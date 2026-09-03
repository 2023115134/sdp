import pytest

from app.crypto.aead import AEADError, decrypt, encrypt
from app.crypto.aead_mapping import (
    aead_to_character_sequence,
    character_sequence_to_aead,
    decrypt_aead_from_mapped,
)
from app.crypto.key_derivation import derive_keys
from app.crypto.mapping import CharacterMap
from app.crypto.position_generator import generate_positions
from app.extraction.extractor import Extractor
from app.llm.embedder import EmbedderLLM


def test_receiver_decryption_with_correct_nonce_and_key():
    plaintext = b"receiver decryption works"
    dk1, _ = derive_keys("correct horse battery staple", b"\x01\x02\x03\x04\x05\x06\x07\x08")
    encrypted = encrypt(plaintext, dk1)

    recovered = decrypt(
        ciphertext=encrypted["ciphertext"],
        tag=encrypted["tag"],
        nonce=encrypted["nonce"],
        dk1=dk1,
    )

    assert recovered == plaintext


def test_receiver_decryption_rejects_wrong_key():
    plaintext = b"wrong key"
    dk1, _ = derive_keys("correct horse battery staple", b"\x01\x02\x03\x04\x05\x06\x07\x08")
    wrong_key = bytes((b + 1) % 256 for b in dk1)
    encrypted = encrypt(plaintext, dk1)

    with pytest.raises((AEADError, ValueError)):
        decrypt(
            ciphertext=encrypted["ciphertext"],
            tag=encrypted["tag"],
            nonce=encrypted["nonce"],
            dk1=wrong_key,
        )


def test_receiver_decryption_rejects_modified_ciphertext():
    plaintext = b"tampered ciphertext"
    dk1, _ = derive_keys("correct horse battery staple", b"\x01\x02\x03\x04\x05\x06\x07\x08")
    encrypted = encrypt(plaintext, dk1)

    tampered_ciphertext = bytearray(encrypted["ciphertext"])
    if tampered_ciphertext:
        tampered_ciphertext[0] ^= 0xFF

    with pytest.raises((AEADError, ValueError)):
        decrypt(
            ciphertext=bytes(tampered_ciphertext),
            tag=encrypted["tag"],
            nonce=encrypted["nonce"],
            dk1=dk1,
        )


def test_receiver_decryption_rejects_modified_tag():
    plaintext = b"tampered tag"
    dk1, _ = derive_keys("correct horse battery staple", b"\x01\x02\x03\x04\x05\x06\x07\x08")
    encrypted = encrypt(plaintext, dk1)

    tampered_tag = bytearray(encrypted["tag"])
    tampered_tag[0] ^= 0xFF

    with pytest.raises((AEADError, ValueError)):
        decrypt(
            ciphertext=encrypted["ciphertext"],
            tag=bytes(tampered_tag),
            nonce=encrypted["nonce"],
            dk1=dk1,
        )


def test_receiver_decryption_rejects_wrong_nonce():
    plaintext = b"wrong nonce"
    dk1, _ = derive_keys("correct horse battery staple", b"\x01\x02\x03\x04\x05\x06\x07\x08")
    encrypted = encrypt(plaintext, dk1)
    wrong_nonce = bytes((b + 1) % 256 for b in encrypted["nonce"])

    with pytest.raises((AEADError, ValueError)):
        decrypt(
            ciphertext=encrypted["ciphertext"],
            tag=encrypted["tag"],
            nonce=wrong_nonce,
            dk1=dk1,
        )


def test_full_receiver_pipeline_round_trip(monkeypatch):
    password = "correct horse battery staple"
    salt = b"\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11"
    dk1, dk2 = derive_keys(password, salt)
    plaintext = b"hello receiver"

    encrypted = encrypt(plaintext, dk1)
    mapped = aead_to_character_sequence(encrypted["enc"])

    positions = generate_positions(
        key_material=dk2,
        number_of_positions=len(mapped),
        offset_do=32,
        max_story_length=20000,
        min_gap=1,
    )

    def fake_embed_one_character(self, story, character, position, topic, temperature, top_k, max_retries, max_steps, max_attempts, attempt_counter):
        if len(story) <= position:
            story += " " * (position - len(story) + 1)
        return story[:position] + character + story[position + 1 :]

    monkeypatch.setattr(EmbedderLLM, "_embed_one_character", fake_embed_one_character)

    embedder = EmbedderLLM(character_map=CharacterMap())
    result = embedder.embed(
        topic="forest trail",
        characters=mapped,
        positions=positions,
        initial_story="A quiet trail winds through the forest.",
        temperature=0.7,
        top_k=40,
        max_new_tokens=16,
        max_attempts=100,
        max_retries=2,
    )

    extractor = Extractor(character_map=CharacterMap())
    extracted = extractor.extract(
        cover_text=result.story,
        positions=positions,
    )

    assert extracted == mapped
    enc_prime = character_sequence_to_aead(extracted)
    assert enc_prime == encrypted["enc"]

    recovered_plaintext = decrypt_aead_from_mapped(extracted, encrypted["nonce"], dk1)
    assert recovered_plaintext == plaintext
