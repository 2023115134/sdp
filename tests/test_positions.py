import pytest

from app.crypto.position_generator import PositionGenerator


def test_position_reproducibility():
    generator = PositionGenerator(chunk_size=5, offset_do=32, max_story_length=1000)
    positions_a = generator.generate("secret-key", 20)
    positions_b = generator.generate("secret-key", 20)
    assert positions_a == positions_b


def test_different_keys_produce_different_positions():
    generator = PositionGenerator(chunk_size=5, offset_do=32, max_story_length=2000)
    positions_a = generator.generate("key-a", 20)
    positions_b = generator.generate("key-b", 20)
    assert positions_a != positions_b


def test_position_ordering_is_sorted():
    generator = PositionGenerator(chunk_size=5, offset_do=32, max_story_length=5000)
    positions = generator.generate("demo-key", 15)
    assert positions == sorted(positions)
    assert len(positions) == len(set(positions))
