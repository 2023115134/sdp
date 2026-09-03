import pytest

from app.crypto.aead import encrypt
from app.crypto.aead_mapping import (
    aead_to_character_sequence,
    character_sequence_to_aead,
)
from app.crypto.key_derivation import derive_keys
from app.crypto.mapping import CharacterMap


@pytest.mark.parametrize(
    "plaintext",
    [
        b"",
        b"A",
        b"hello",
        b"hello world",
        b"phase-2-aead-mapping",
        bytes(range(64)),
    ],
)
def test_aead_mapping_round_trip(plaintext):
    password = "correct horse battery staple"
    salt = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    dk1, _ = derive_keys(password, salt)

    encrypted = encrypt(plaintext, dk1)
    enc = encrypted["enc"]

    mapped = aead_to_character_sequence(enc)

    assert isinstance(mapped, str)
    assert len(mapped) == 2 * len(enc)
    assert all(character in CharacterMap.ALPHABET for character in mapped)
    assert mapped == aead_to_character_sequence(enc)

    recovered_enc = character_sequence_to_aead(mapped)
    assert recovered_enc == enc


def test_aead_mapping_maps_each_byte_to_two_nibbles():
    enc = bytes((0x00, 0x12, 0xAB, 0xFF))
    character_map = CharacterMap()

    mapped = aead_to_character_sequence(enc)
    expected = "".join(
        character_map.VALUE_TO_CHAR[nibble]
        for byte in enc
        for nibble in ((byte >> 4) & 0x0F, byte & 0x0F)
    )

    assert mapped == expected
    assert len(mapped) == 2 * len(enc)
    assert character_sequence_to_aead(mapped) == enc


def test_aead_mapping_rejects_odd_length_sequence():
    with pytest.raises(ValueError, match="even number"):
        character_sequence_to_aead(" E ")


def test_nonce_is_preserved_separately():
    password = "correct horse battery staple"
    salt = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    dk1, _ = derive_keys(password, salt)

    encrypted = encrypt(b"keep nonce separate", dk1)

    assert encrypted["nonce"]
    assert len(encrypted["nonce"]) == 12
    assert encrypted["enc"] == encrypted["tag"] + encrypted["ciphertext"]


def test_mapping_output_uses_existing_character_alphabet():
    password = "correct horse battery staple"
    salt = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    dk1, _ = derive_keys(password, salt)

    encrypted = encrypt(b"mapping alphabet check", dk1)
    mapped = aead_to_character_sequence(encrypted["enc"])

    assert set(mapped).issubset(set(CharacterMap.ALPHABET))
    assert mapped
