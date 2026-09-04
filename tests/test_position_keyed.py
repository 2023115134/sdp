import pytest

from app.crypto.key_derivation import derive_keys
from app.crypto.position_generator import PositionGenerator, generate_positions


def test_same_dk2_same_parameters_produce_same_positions():
    password = "correct horse battery staple"
    salt = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    _, dk2 = derive_keys(password, salt)

    positions_1 = generate_positions(
        key_material=dk2,
        number_of_positions=12,
        offset_do=32,
        max_story_length=5000,
        min_gap=10,
    )
    positions_2 = generate_positions(
        key_material=dk2,
        number_of_positions=12,
        offset_do=32,
        max_story_length=5000,
        min_gap=10,
    )

    assert positions_1 == positions_2


def test_different_dk2_same_parameters_produce_different_positions():
    password_a = "correct horse battery staple"
    password_b = "correct horse battery staple!"
    salt = b"\x01\x02\x03\x04\x05\x06\x07\x08"

    _, dk2_a = derive_keys(password_a, salt)
    _, dk2_b = derive_keys(password_b, salt)

    positions_a = generate_positions(
        key_material=dk2_a,
        number_of_positions=12,
        offset_do=32,
        max_story_length=5000,
        min_gap=10,
    )
    positions_b = generate_positions(
        key_material=dk2_b,
        number_of_positions=12,
        offset_do=32,
        max_story_length=5000,
        min_gap=10,
    )

    assert positions_a != positions_b


def test_dk2_must_be_bytes_accepted():
    _, dk2 = derive_keys("correct horse battery staple", b"\x10\x11\x12\x13")
    positions = generate_positions(key_material=dk2, number_of_positions=5, offset_do=32, max_story_length=5000, min_gap=10)
    assert isinstance(positions, list)
    assert len(positions) == 5


def test_32_byte_dk2_works():
    _, dk2 = derive_keys("correct horse battery staple", b"\x01\x02\x03\x04\x05\x06\x07\x08")
    assert len(dk2) == 32
    positions = generate_positions(key_material=dk2, number_of_positions=8, offset_do=32, max_story_length=2000, min_gap=5)
    assert len(positions) == 8
    assert positions == sorted(positions)


def test_invalid_key_material_rejected():
    with pytest.raises((TypeError, ValueError)):
        generate_positions(key_material=b"", number_of_positions=5, offset_do=32, max_story_length=5000, min_gap=5)

    with pytest.raises((TypeError, ValueError)):
        generate_positions(key_material="", number_of_positions=5, offset_do=32, max_story_length=5000, min_gap=5)

    with pytest.raises((TypeError, ValueError)):
        generate_positions(key_material=None, number_of_positions=5, offset_do=32, max_story_length=5000, min_gap=5)


def test_existing_phase1_behavior_still_passes():
    generator = PositionGenerator(min_gap=5, offset_do=32, max_story_length=1000)
    positions_a = generator.generate(key_material="secret-key", number_of_positions=20)
    positions_b = generator.generate(key_material="secret-key", number_of_positions=20)
    assert positions_a == positions_b


def test_offset_and_min_gap_behavior_remains_consistent():
    generator = PositionGenerator(min_gap=5, offset_do=32, max_story_length=1000)
    positions = generator.generate(key_material="demo-key", number_of_positions=15)
    assert positions == sorted(positions)
    assert len(positions) == len(set(positions))
    assert all(position >= 32 for position in positions)


def test_generated_positions_remain_valid_under_constraints():
    _, dk2 = derive_keys("check key", b"\x0a\x0b\x0c\x0d")
    positions = generate_positions(key_material=dk2, number_of_positions=10, offset_do=32, max_story_length=2000, min_gap=7)
    assert len(positions) == 10
    assert all(32 <= pos < 2000 for pos in positions)
    assert all(positions[i + 1] - positions[i] >= 7 for i in range(len(positions) - 1))
