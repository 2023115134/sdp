from app.crypto.aead import encrypt
from app.crypto.aead_mapping import aead_to_character_sequence, character_sequence_to_aead
from app.crypto.key_derivation import derive_keys
from app.crypto.mapping import CharacterMap
from app.crypto.position_generator import generate_positions
from app.extraction.extractor import Extractor
from app.llm.embedder import EmbedderLLM


def test_secure_embedding_round_trip_through_embedder(monkeypatch):
    password = "correct horse battery staple"
    salt = b"\x11\x22\x33\x44\x55\x66\x77\x88"

    dk1, dk2 = derive_keys(password, salt)

    plaintext = b"hello"
    encrypted = encrypt(plaintext, dk1)
    enc = encrypted["enc"]
    mapped = aead_to_character_sequence(enc)

    positions = generate_positions(
        key_material=dk2,
        number_of_positions=len(mapped),
        offset_do=32,
        max_story_length=5000,
        min_gap=1,
    )

    assert len(mapped) == len(positions)
    assert len(enc) > 0

    def fake_embed_one_character(self, story, character, position, topic, temperature, top_k, max_retries, max_steps, max_attempts, attempt_counter):
        filler_len = max(0, position - len(story))
        if filler_len:
            story += " " * filler_len
        if len(story) <= position:
            story += "x"
        story = story[:position] + character + story[position + 1 :]
        return story

    monkeypatch.setattr(EmbedderLLM, "_embed_one_character", fake_embed_one_character)

    embedder = EmbedderLLM(character_map=CharacterMap())
    result = embedder.embed(
        topic="forest trail",
        characters=mapped,
        positions=positions,
        initial_story="A quiet forest trail winds through the valley.",
        temperature=0.7,
        top_k=40,
        max_new_tokens=32,
        max_attempts=100,
        max_retries=2,
    )

    assert result.story
    assert result.embedded_characters == mapped

    extractor = Extractor(character_map=CharacterMap())
    extracted = extractor.extract(
        cover_text=result.story,
        positions=positions,
    )

    assert extracted == mapped
    recovered_enc = character_sequence_to_aead(extracted)
    assert recovered_enc == enc

    assert result.story != ""
