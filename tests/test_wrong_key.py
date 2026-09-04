import pytest

from app.crypto.mapping import CharacterMap
from app.crypto.position_generator import PositionGenerator
from app.extraction.extractor import Extractor


def test_key_dependent_extraction_validation():
    mapping = CharacterMap()
    generator = PositionGenerator(offset_do=32, max_story_length=2000)
    key = "test-secret-key"
    hidden = mapping.encode("HELLO")
    positions = generator.generate_for_message(
        key_material=key,
        message_length=len(hidden),
    )

    cover = ["x"] * (max(positions) + 1)
    for position, character in zip(positions, hidden):
        cover[position] = character

    extractor = Extractor(generator, mapping)
    assert extractor.recover("".join(cover), key, len(hidden)).message == "HELLO"

    try:
        wrong = extractor.recover("".join(cover), "wrong-key", len(hidden)).message
    except ValueError:
        return

    assert wrong != "HELLO"
