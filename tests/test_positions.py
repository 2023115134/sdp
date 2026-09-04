import pytest

from app.crypto.position_generator import PositionGenerator


def test_position_reproducibility():
    generator = PositionGenerator(min_gap=5, offset_do=32, max_story_length=1000)
    positions_a = generator.generate(key_material="secret-key", number_of_positions=20)
    positions_b = generator.generate(key_material="secret-key", number_of_positions=20)
    assert positions_a == positions_b


def test_different_keys_produce_different_positions():
    generator = PositionGenerator(min_gap=5, offset_do=32, max_story_length=2000)
    positions_a = generator.generate(key_material="key-a", number_of_positions=20)
    positions_b = generator.generate(key_material="key-b", number_of_positions=20)
    assert positions_a != positions_b


def test_position_ordering_is_sorted():
    generator = PositionGenerator(min_gap=5, offset_do=32, max_story_length=5000)
    positions = generator.generate(key_material="demo-key", number_of_positions=15)
    assert positions == sorted(positions)
    assert len(positions) == len(set(positions))
