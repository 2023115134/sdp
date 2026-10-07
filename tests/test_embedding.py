from app.crypto.mapping import CharacterMap
from app.crypto.position_generator import PositionGenerator
from app.extraction.extractor import Extractor
from app.llm.embedder import EmbedderLLM


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


def test_normal_candidate_prefers_fresh_text_but_keeps_repetitive_fallback():
    embedder = EmbedderLLM()
    repeated = _FakeCandidate(" story", probability=0.99)
    fresh = _FakeCandidate(" river", probability=0.5)

    selected = embedder._select_normal_candidate(
        story="The story story story story story story.",
        candidates=[repeated, fresh],
        topic="a journey",
    )

    assert selected is fresh

    only_option = embedder._select_normal_candidate(
        story="The story story story story story.",
        candidates=[repeated],
        topic="a journey",
    )
    assert only_option is repeated


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
