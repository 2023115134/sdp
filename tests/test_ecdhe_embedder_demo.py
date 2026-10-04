import pytest

from app.crypto.aead import decrypt as aead_decrypt, encrypt as aead_encrypt
from app.crypto.ecdhe import derive_session_keys, generate_key_pair
from app.crypto.ecdhe_embedder_demo import (
    decode_public_key_from_embedding,
    extract_public_key_from_cover_text,
    public_key_to_embeddable_text,
)
from app.crypto.mapping import CharacterMap
from app.crypto.position_generator import generate_positions
from app.demo import _build_safe_initial_story
from app.extraction.extractor import Extractor
from app.llm.embedder import EmbedderLLM, EmbeddingResult


def _build_cover_with_payload(initial_story, characters, positions):
    story = initial_story
    for character, position in zip(characters, positions):
        if position < len(story):
            story = story[:position] + character + story[position + 1 :]
        else:
            story = story.ljust(position, " ") + character
    return story


def _fake_embed(self, topic, characters, positions, initial_story="", temperature=0.7, top_k=40, max_new_tokens=32, max_attempts=100, max_retries=2, deterministic=False):
    story = _build_cover_with_payload(initial_story or topic, characters, positions)
    return EmbeddingResult(
        story=story,
        embedded_characters=characters,
        positions=list(positions),
        attempts=0,
    )


def test_public_key_encoding_and_mapping_round_trip():
    _, public_key = generate_key_pair()
    encoded = public_key_to_embeddable_text(public_key)

    assert isinstance(encoded, str)
    assert encoded == encoded.lower()
    assert bytes.fromhex(encoded) == public_key
    assert decode_public_key_from_embedding(encoded) == public_key

    mapped = CharacterMap().encode(encoded)
    decoded = CharacterMap().decode(mapped)
    assert decoded == encoded


def test_safe_initial_story_stays_before_first_target_position():
    positions = [64, 93, 120, 170]
    story = _build_safe_initial_story("A quiet city street at dusk with bright windows", positions)

    assert 0 < len(story) < positions[0]
    assert story.strip()


def test_public_key_embedding_and_extraction_round_trip(monkeypatch):
    _, public_key = generate_key_pair()
    encoded = public_key_to_embeddable_text(public_key)
    mapped = CharacterMap().encode(encoded)
    positions = generate_positions(
        key_material=b"ecdhe-demo-key-material",
        number_of_positions=len(mapped),
        offset_do=32,
        max_story_length=20000,
        min_gap=1,
    )

    monkeypatch.setattr(EmbedderLLM, "embed", _fake_embed)

    embedder = EmbedderLLM(character_map=CharacterMap())
    result = embedder.embed(
        topic="city streets at dusk",
        characters=mapped,
        positions=positions,
        initial_story="A quiet city street glows softly at dusk.",
        temperature=0.7,
        top_k=40,
        max_new_tokens=32,
        max_attempts=200,
        max_retries=2,
    )

    extracted = Extractor.extract(result.story, positions)
    decoded = CharacterMap().decode(extracted)
    assert decoded == encoded
    assert decode_public_key_from_embedding(decoded) == public_key
    assert extract_public_key_from_cover_text(result.story, positions) == public_key


def test_modified_carrier_text_rejects_public_key_recovery():
    _, public_key = generate_key_pair()
    encoded = public_key_to_embeddable_text(public_key)
    mapped = CharacterMap().encode(encoded)
    positions = generate_positions(
        key_material=b"tamper-demo-key",
        number_of_positions=len(mapped),
        offset_do=32,
        max_story_length=20000,
        min_gap=1,
    )

    base_story = "A quiet market square leads into the evening light."
    cover = _build_cover_with_payload(base_story, mapped, positions)

    corrupted = cover[:positions[0]] + ("E" if cover[positions[0]] != "E" else "T") + cover[positions[0] + 1 :]
    with pytest.raises((ValueError, TypeError)):
        extract_public_key_from_cover_text(corrupted, positions)

    wrong_positions = generate_positions(
        key_material=b"different-key-material",
        number_of_positions=len(mapped),
        offset_do=32,
        max_story_length=20000,
        min_gap=1,
    )
    with pytest.raises(ValueError):
        extract_public_key_from_cover_text(cover, wrong_positions)


def test_complete_ecdhe_public_key_to_aes_round_trip(monkeypatch):
    alice_private, alice_public = generate_key_pair()
    bob_private, bob_public = generate_key_pair()
    salt = b"shared-salt-ecdhe-1"[:16].ljust(16, b"0")
    context = b"carrier-demo-session"

    encoded = public_key_to_embeddable_text(alice_public)
    mapped = CharacterMap().encode(encoded)
    positions = generate_positions(
        key_material=b"carrier-session-key",
        number_of_positions=len(mapped),
        offset_do=32,
        max_story_length=20000,
        min_gap=1,
    )

    monkeypatch.setattr(EmbedderLLM, "embed", _fake_embed)

    embedder = EmbedderLLM(character_map=CharacterMap())
    result = embedder.embed(
        topic="friends meeting on a rainy evening",
        characters=mapped,
        positions=positions,
        initial_story="The train arrived just as the rain began to fall.",
        temperature=0.7,
        top_k=40,
        max_new_tokens=32,
        max_attempts=200,
        max_retries=2,
    )

    recovered_public = extract_public_key_from_cover_text(result.story, positions)
    assert recovered_public == alice_public

    alice_send, _ = derive_session_keys(alice_private, bob_public, salt, context)
    _, bob_receive = derive_session_keys(bob_private, recovered_public, salt, context)
    assert alice_send == bob_receive

    message = b"secret from Alice through the carrier"
    packet = aead_encrypt(message, alice_send, context)
    plaintext = aead_decrypt(
        packet["ciphertext"],
        packet["tag"],
        packet["nonce"],
        bob_receive,
        context,
    )
    assert plaintext == message
