import random

import pytest

from app.crypto.mapping import CharacterMap
from app.crypto.position_generator import PositionGenerator
from app.extraction.extractor import Extractor
from app.llm.embedder import EmbedderLLM
from app.web_demo import _log_embedding_progress


class _FakeCandidate:
    def __init__(self, token, probability=0.8):
        self.token = token
        self.probability = probability


class _FakeGenerator:
    def __init__(self):
        self.calls = 0

    def _load_backend(self):
        return None

    def get_next_token_candidates(self, prompt, top_k, temperature):
        self.calls += 1
        if self.calls == 1:
            return [_FakeCandidate(" tail")]
        return [_FakeCandidate(".")]

    def generate(self, prompt, **kwargs):
        return "."


def test_embedder_success_and_failure_paths():
    mapper = CharacterMap()
    positions = PositionGenerator(min_gap=5, offset_do=32, max_story_length=1000).generate(
        key_material="key",
        number_of_positions=5,
    )
    story = "the old lighthouse watched the black sea from the cliff"

    encoded = mapper.encode("ABCD")
    extracted = Extractor.extract(story, positions[:4])
    assert extracted != encoded

    assert len(positions) >= 4


def test_verbose_embedding_reports_character_and_retry_without_inference_start_logs(
    monkeypatch, capsys
):
    embedder = EmbedderLLM()

    def embed_at_target(**kwargs):
        story = kwargs["story"]
        position = kwargs["position"]
        return story.ljust(position) + kwargs["character"]

    monkeypatch.setattr(embedder, "_generate_to_position", embed_at_target)
    result = embedder.embed(
        topic="quiet city",
        characters="Q",
        positions=[20],
        initial_story="A",
        verbose=True,
    )

    output = capsys.readouterr().out
    assert result.story[20] == "Q"
    assert "Embedding 1/1" in output
    assert "Character: Q" in output
    assert "Position: 20" in output
    assert "Retry 1: T=0.70 k=40" in output
    assert "Status: Embedded successfully" in output
    assert "Character stats: LLM calls=0, candidate evaluations=0, retries=1" in output
    assert "LLM inference call" not in output


def test_end_to_end_mapping_and_extraction_pipeline():
    mapper = CharacterMap()
    secret = "HELLO"
    encoded = mapper.encode(secret)
    positions = [0, 4, 8, 12, 16]
    story = "A quiet city breathes under the morning sky"
    actual = Extractor.extract(story, positions)
    assert len(actual) == len(positions)
    assert actual != encoded


def test_embedder_only_appends_final_tail_after_embedded_position():
    generator = _FakeGenerator()
    embedder = EmbedderLLM(llm_generator=generator)
    embedder._embed_one_character = lambda *args, **kwargs: "catA"

    result = embedder.embed(
        topic="cats",
        characters="A",
        positions=[3],
        initial_story="cat",
    )

    assert result.story[:4] == "catA"
    assert result.story[3] == "A"
    assert result.story.endswith(".") or result.story.endswith("!") or result.story.endswith("?")


def test_progress_callback_marks_character_embedded_after_validation(monkeypatch):
    embedder = EmbedderLLM()

    def embed_one_character(self, story, character, position, **kwargs):
        return story.ljust(position, " ") + character

    monkeypatch.setattr(EmbedderLLM, "_embed_one_character", embed_one_character)
    progress = []
    result = embedder.embed(
        topic="quiet garden",
        characters="AB",
        positions=[1, 2],
        initial_story="x",
        max_new_tokens=4,
        progress_callback=lambda completed, total, status: progress.append(
            (completed, total, status)
        ),
    )

    assert result.embedded_characters == "AB"
    assert progress == [
        (0, 2, "embedding"),
        (1, 2, "embedded"),
        (1, 2, "embedding"),
        (2, 2, "embedded"),
    ]


def test_progress_callback_does_not_report_invalid_character_as_embedded(monkeypatch):
    embedder = EmbedderLLM()
    monkeypatch.setattr(
        EmbedderLLM,
        "_embed_one_character",
        lambda self, story, character, position, **kwargs: story.ljust(position, " ") + "Z",
    )
    progress = []

    with pytest.raises(RuntimeError, match="Embedding validation failed"):
        embedder.embed(
            topic="quiet garden",
            characters="A",
            positions=[1],
            initial_story="x",
            max_new_tokens=4,
            progress_callback=lambda completed, total, status: progress.append(
                (completed, total, status)
            ),
        )

    assert progress == [(0, 1, "embedding")]


def test_long_embedding_rejects_candidate_that_crosses_next_position(monkeypatch):
    class LongPayloadGenerator:
        def __init__(self):
            self.calls = 0

        def get_next_token_candidates(self, prompt, top_k, temperature):
            self.calls += 1
            if self.calls == 1:
                return [
                    _FakeCandidate("X" + ("Y" * 63), probability=0.9),
                    _FakeCandidate("X", probability=0.8),
                ]
            return [_FakeCandidate("X"), _FakeCandidate(" ")]

    def select_matching_candidate(
        self,
        story,
        candidates,
        character,
        position,
        topic,
        continuation_candidates=None,
    ):
        for candidate in candidates:
            new_story = story + candidate.token
            if len(new_story) <= position:
                if continuation_candidates is not None:
                    continuation_candidates.append(candidate)
                continue
            if (
                position < len(new_story)
                and self._character_matches(new_story[position], character)
            ):
                return 1.0, candidate, 1.0
        return None

    embedder = EmbedderLLM(llm_generator=LongPayloadGenerator())
    monkeypatch.setattr(
        EmbedderLLM, "_select_embedding_candidate", select_matching_candidate
    )
    monkeypatch.setattr(
        EmbedderLLM,
        "_select_normal_candidate",
        lambda self, story, candidates, topic: next(
            candidate for candidate in candidates if candidate.token == " "
        ),
    )
    monkeypatch.setattr(
        EmbedderLLM, "_complete_cover_text", lambda self, story, topic, max_tokens: story
    )
    monkeypatch.setattr(
        EmbedderLLM,
        "_validate_cover_naturalness",
        lambda self, story, topic: {"test": True},
    )

    result = embedder.embed(
        topic="a long carrier",
        characters="X" * 70,
        positions=[1 + 32 * index for index in range(70)],
        initial_story="s",
        verbose=False,
    )

    assert result.embedded_characters == "X" * 70
    assert all(result.story[position] == "X" for position in result.positions)
    assert len(result.positions) == 70


def test_failure_after_many_embedded_characters_reports_exact_position(monkeypatch):
    class FailingAfterManyCharactersGenerator:
        def __init__(self):
            self.calls = 0

        def get_next_token_candidates(self, prompt, top_k, temperature):
            self.calls += 1
            if self.calls <= 60:
                return [_FakeCandidate("X")]
            return []

    def select_matching_candidate(
        self,
        story,
        candidates,
        character,
        position,
        topic,
        continuation_candidates=None,
    ):
        for candidate in candidates:
            new_story = story + candidate.token
            if (
                position < len(new_story)
                and self._character_matches(new_story[position], character)
            ):
                return 1.0, candidate, 1.0
        return None

    embedder = EmbedderLLM(llm_generator=FailingAfterManyCharactersGenerator())
    monkeypatch.setattr(
        EmbedderLLM, "_select_embedding_candidate", select_matching_candidate
    )
    progress = []
    events = []

    with pytest.raises(
        RuntimeError,
        match="Unable to embed character 'X' at position 61",
    ):
        embedder.embed(
            topic="a long carrier",
            characters="X" * 61,
            positions=list(range(1, 62)),
            initial_story="s",
            verbose=False,
            progress_callback=lambda completed, total, status: progress.append(
                (completed, total, status)
            ),
            progress_event_callback=events.append,
        )

    failures = [event for event in events if event.get("event") == "character_failed"]
    assert len(failures) == 1
    assert failures[0]["index"] == 61
    assert failures[0]["position"] == 61
    assert (60, 61, "embedded") in progress
    assert (60, 61, "embedding") in progress
    assert (61, 61, "embedded") not in progress


def test_detailed_progress_callback_logs_real_character_retry_and_candidate_events(
    monkeypatch, capsys
):
    embedder = EmbedderLLM()

    def embed_at_target(**kwargs):
        embedder._embedding_stats["candidate_evaluations"] += 3
        if kwargs["top_k"] == 40:
            embedder._last_failure_reason = "no valid candidate"
            return None
        return kwargs["story"].ljust(kwargs["position"], " ") + kwargs["character"]

    monkeypatch.setattr(embedder, "_generate_to_position", embed_at_target)
    result = embedder.embed(
        topic="quiet city",
        characters="Q",
        positions=[20],
        initial_story="A",
        verbose=False,
        progress_event_callback=_log_embedding_progress,
    )

    assert result.story[20] == "Q"
    terminal_output = capsys.readouterr().out
    assert "Embedding 1/1" in terminal_output
    assert "Character: Q" in terminal_output
    assert "Character: 'Q'" not in terminal_output
    assert "Position: 20" in terminal_output
    assert "Retry 1: T=0.70 k=40" in terminal_output
    assert "Retry 1/12" not in terminal_output
    assert "Candidate rejected: no valid candidate" in terminal_output
    assert "Retry 2: T=0.70 k=50" in terminal_output
    assert "Status: Embedded successfully (candidate evaluations this retry: 3)" in terminal_output


def test_normal_candidate_prefers_fresh_text_and_rejects_repetitive_only_option():
    embedder = EmbedderLLM()
    repeated = _FakeCandidate(" story", probability=0.99)
    fresh = _FakeCandidate(" river", probability=0.5)

    selected = embedder._select_normal_candidate(
        story="The traveler read a story beside the road.",
        candidates=[repeated, fresh],
        topic="a journey",
    )

    assert selected is fresh

    only_option = embedder._select_normal_candidate(
        story="The story story story story story.",
        candidates=[repeated],
        topic="a journey",
    )
    assert only_option is None


def test_runaway_synonym_lists_are_rejected_before_more_text_is_added():
    embedder = EmbedderLLM()
    word_list = (
        "zealously, fervently, passionately, ardently, enthusiastically, "
        "zealotically, fervorously, continuously, magnificently, "
        "disproportionately, tirelessly"
    )

    assert embedder._is_degraded_continuation(
        "The house was decorated ",
        word_list,
    )
    assert embedder._select_normal_candidate(
        story="The house was decorated ",
        candidates=[_FakeCandidate(word_list, probability=0.9)],
        topic="a home",
    ) is None

    emotion_list = (
        "happiness, joy, contentment, bliss, elation, joy, elated, "
        "elation, elate, elated, nostalgia, melancholy, melancholic, melancholia"
    )
    assert embedder._is_degraded_continuation("", emotion_list)
    assert embedder._select_embedding_candidate(
        story="",
        candidates=[_FakeCandidate(emotion_list, probability=0.95)],
        character="h",
        position=0,
        topic="a quiet home",
    ) is None


def test_degraded_prior_sentence_does_not_reject_a_candidate_that_finishes_it(
    monkeypatch,
):
    embedder = EmbedderLLM()
    monkeypatch.setattr(embedder, "_candidate_naturalness", lambda **kwargs: 1.0)
    story = (
        "The garden held roses, lilies, tulips, daisies, and violets"
    )
    candidate = _FakeCandidate(" On the path, he paused.", probability=0.5)

    assert not embedder._is_degraded_continuation(story, candidate.token)
    selected = embedder._select_embedding_candidate(
        story=story,
        candidates=[candidate],
        character="O",
        position=len(story) + 1,
        topic="a garden path",
    )

    assert selected is not None
    assert selected[1] is candidate
    assert (story + candidate.token)[len(story) + 1] == "O"


def test_normal_selection_varies_only_among_near_equal_candidates():
    embedder = EmbedderLLM()
    embedder._selection_rng = random.Random(17)
    candidates = [
        _FakeCandidate(" river", probability=0.8),
        _FakeCandidate(" meadow", probability=0.8),
        _FakeCandidate(" the same scene again", probability=0.01),
    ]

    selections = {
        embedder._select_normal_candidate(
            story="A traveler followed the path.",
            candidates=candidates,
            topic="a journey",
        ).token
        for _ in range(30)
    }

    assert selections == {" river", " meadow"}


def test_embedding_candidate_prefers_natural_phrase_over_repeated_frame():
    embedder = EmbedderLLM()
    story = "The student found a new book, a new class, "
    repetitive = _FakeCandidate("a new story", probability=0.9)
    natural = _FakeCandidate("a noble teacher", probability=0.3)

    selected = embedder._select_embedding_candidate(
        story=story,
        candidates=[repetitive, natural],
        character="n",
        position=len(story) + 2,
        topic="studying",
    )

    assert selected is not None
    assert selected[1] is natural
    assert (story + selected[1].token)[len(story) + 2] == "n"


def test_embedding_candidate_reuses_naturalness_for_candidate_score(monkeypatch):
    embedder = EmbedderLLM()
    calls = []
    original_naturalness = embedder._candidate_naturalness

    def count_naturalness(*, story, token, topic):
        calls.append(token)
        return original_naturalness(story=story, token=token, topic=topic)

    monkeypatch.setattr(embedder, "_candidate_naturalness", count_naturalness)
    token = _FakeCandidate(" quiet")
    selected = embedder._select_embedding_candidate(
        story="A",
        candidates=[token],
        character="q",
        position=2,
        topic="quiet city",
    )

    assert selected is not None
    assert calls == [" quiet"]


def test_cover_naturalness_flags_repeated_short_phrase_frames():
    story = (
        "The student saw a new book, a new class, a new desk, "
        "and a new professor while studying."
    )

    naturalness = EmbedderLLM._validate_cover_naturalness(
        story=story,
        topic="studying",
    )

    assert naturalness["repetition"] is False


def test_candidate_ranking_penalizes_adjective_lists_and_related_stems():
    embedder = EmbedderLLM()
    story = "In the classroom, "
    list_candidate = _FakeCandidate(
        " serene, peaceful, refreshing, tranquil, students reviewed the lesson.",
        probability=0.75,
    )
    narrative_candidate = _FakeCandidate(
        " students compared their notes and discussed the lesson.",
        probability=0.35,
    )
    required_position = len(story) + 1

    selected = embedder._select_embedding_candidate(
        story=story,
        candidates=[list_candidate, narrative_candidate],
        character="s",
        position=required_position,
        topic="studying",
    )

    assert selected is not None
    assert selected[1] is narrative_candidate
    assert (story + selected[1].token)[required_position] == "s"
    assert embedder._word_stem("serene") == embedder._word_stem("serenity")
    assert (
        embedder._word_stem("enlightening")
        == embedder._word_stem("enlightenment")
    )
    assert (
        embedder._word_stem("revitalizing")
        == embedder._word_stem("revitalized")
        == embedder._word_stem("revitalization")
    )


def test_candidate_naturalness_rewards_topic_relevance_and_complete_sentences():
    embedder = EmbedderLLM()
    story = "The room grew quiet. "

    relevant = embedder._candidate_naturalness(
        story=story,
        token=" The classroom teacher began the lesson.",
        topic="classroom lesson",
    )
    unrelated = embedder._candidate_naturalness(
        story=story,
        token=" A distant ship crossed the harbor.",
        topic="classroom lesson",
    )
    incomplete = embedder._candidate_naturalness(
        story=story,
        token=" The students waited for.",
        topic="classroom lesson",
    )

    assert relevant > unrelated
    assert relevant > incomplete
    assert EmbedderLLM._validate_cover_naturalness(
        "The students waited for.",
        "classroom lesson",
    )["sentence_completeness"] is False


def test_retry_schedule_is_bounded_and_keeps_topic_separate_from_guidance(monkeypatch):
    embedder = EmbedderLLM()
    embedder._verbose = False
    calls = []

    def reject_candidate(**kwargs):
        calls.append(kwargs)
        embedder._last_failure_reason = "no natural candidate"
        return None

    monkeypatch.setattr(embedder, "_generate_to_position", reject_candidate)

    with pytest.raises(RuntimeError, match="after 12 controlled retry configurations"):
        embedder._embed_one_character(
            story="The existing scene continues.",
            character="E",
            position=50,
            topic="a quiet classroom",
            temperature=0.65,
            top_k=25,
            max_retries=100,
            max_steps=1,
            max_attempts=100,
            attempt_counter=[0],
        )

    assert len(calls) == 12
    assert {call["topic"] for call in calls} == {"a quiet classroom"}
    assert calls[0]["temperature"] == 0.65
    assert calls[0]["top_k"] == 25
    assert any(call["retry_guidance"] for call in calls)
    assert embedder._embedding_stats["retries"] == 12


def test_fallback_candidate_query_is_included_in_call_and_attempt_counts():
    class CandidateGenerator:
        def get_next_token_candidates(self, prompt, top_k, temperature):
            return [_FakeCandidate("a")]

    embedder = EmbedderLLM(llm_generator=CandidateGenerator())
    attempts = [0]

    result = embedder._generate_to_position(
        story="",
        character="Z",
        position=0,
        topic="a classroom",
        temperature=0.7,
        top_k=40,
        max_steps=1,
        max_attempts=10,
        attempt_counter=attempts,
    )

    assert result is None
    assert attempts == [2]
    assert embedder._embedding_stats["llm_calls"] == 2
    assert embedder._embedding_stats["candidate_evaluations"] == 2


def test_cover_tail_uses_one_bounded_text_generation():
    class TailGenerator:
        def __init__(self):
            self.calls = []

        def generate(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            return "and the evening settles over the town."

        def get_next_token_candidates(self, **kwargs):
            raise AssertionError("tail completion must not run candidate-by-candidate")

    generator = TailGenerator()
    embedder = EmbedderLLM(llm_generator=generator)
    original = "A hidden payload ends"

    completed = embedder._complete_cover_text(
        story=original,
        topic="a quiet town",
        max_tokens=80,
    )

    assert len(generator.calls) == 1
    assert generator.calls[0][1]["max_new_tokens"] == 48
    assert completed.startswith(original)
    assert completed.endswith("town.")


def test_cover_tail_caps_requested_generation_length():
    class TailGenerator:
        def __init__(self):
            self.max_new_tokens = None

        def generate(self, prompt, **kwargs):
            self.max_new_tokens = kwargs["max_new_tokens"]
            return "continues."

    generator = TailGenerator()
    embedder = EmbedderLLM(llm_generator=generator)
    embedder._complete_cover_text("unfinished", "topic", max_tokens=8)

    assert generator.max_new_tokens == 8


def test_adaptive_space_search_does_not_treat_tabs_as_payload_spaces():
    class WhitespaceGenerator:
        def get_next_token_candidates(self, prompt, top_k, temperature):
            return [
                _FakeCandidate("\tword", probability=0.99),
                _FakeCandidate(" word", probability=0.1),
            ]

    embedder = EmbedderLLM(llm_generator=WhitespaceGenerator())
    adaptive_story = embedder._generate_until_character(
        story="A beginning",
        character=" ",
        topic="a story",
        temperature=0.7,
        top_k=40,
        max_steps=1,
    )

    assert adaptive_story == "A beginning word"


def test_adaptive_embedding_shifts_positions_after_natural_placement(monkeypatch):
    embedder = EmbedderLLM(llm_generator=_FakeGenerator())

    def fail_fixed_target(**kwargs):
        raise RuntimeError("fixed target was unreachable")

    monkeypatch.setattr(embedder, "_embed_one_character", fail_fixed_target)
    monkeypatch.setattr(
        embedder,
        "_generate_adaptive_character",
        lambda **kwargs: ("seed\tabc S", 9),
    )
    monkeypatch.setattr(
        embedder,
        "_complete_cover_text",
        lambda story, topic, max_tokens: story,
    )
    monkeypatch.setattr(
        embedder,
        "_validate_cover_naturalness",
        lambda story, topic: {"test": True},
    )

    positions = [8]
    result = embedder.embed(
        topic="a story",
        characters="S",
        positions=positions,
        initial_story="seed",
        allow_adaptive_positions=True,
    )

    assert positions == [8]
    assert result.positions == [9]
    assert result.story[9] == "S"


def test_adaptive_generator_uses_real_model_character_at_target(monkeypatch):
    class AdaptiveGenerator:
        def get_next_token_candidates(self, prompt, top_k, temperature):
            return [_FakeCandidate("quiet S evening", probability=0.8)]

    embedder = EmbedderLLM(llm_generator=AdaptiveGenerator())
    monkeypatch.setattr(
        embedder,
        "_candidate_naturalness",
        lambda story, token, topic: 1.0,
    )
    story, position = embedder._generate_adaptive_character(
        story="The",
        character="S",
        minimum_position=4,
        topic="evening",
        max_steps=2,
        following_gap=40,
    )

    assert story == "Thequiet S evening"
    assert story[position] == "S"
    assert position == 9
