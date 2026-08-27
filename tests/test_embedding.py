from app.crypto.mapping import CharacterMap
from app.crypto.position_generator import PositionGenerator
from app.extraction.extractor import Extractor


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
