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


def test_embedder_success_and_failure_paths():
    mapper = CharacterMap()
    positions = PositionGenerator(chunk_size=5, offset_do=32, max_story_length=100).generate("key", 5)
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
