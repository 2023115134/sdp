"""Demonstrate ECDHE public-key transport through the LLM carrier workflow."""

from __future__ import annotations

from typing import Sequence

from app.crypto.aead import decrypt as aead_decrypt, encrypt as aead_encrypt
from app.crypto.ecdhe import derive_session_keys, generate_key_pair
from app.crypto.mapping import CharacterMap
from app.crypto.position_generator import generate_positions
from app.extraction.extractor import Extractor
from app.llm.embedder import EmbedderLLM
from app.llm.generator import LLMGenerator


def _normalize_public_key(public_key: bytes | bytearray | memoryview) -> bytes:
    if isinstance(public_key, memoryview):
        public_key = public_key.tobytes()
    if not isinstance(public_key, (bytes, bytearray)):
        raise TypeError("public_key must be bytes-like")
    public_key = bytes(public_key)
    if len(public_key) != 32:
        raise ValueError("public_key must be exactly 32 bytes")
    return public_key


def public_key_to_embeddable_text(public_key: bytes | bytearray | memoryview) -> str:
    """Encode a raw X25519 public key into a reversible ASCII representation."""

    key_bytes = _normalize_public_key(public_key)
    return key_bytes.hex()


def decode_public_key_from_embedding(encoded_public_key: str) -> bytes:
    """Recover the original raw key bytes from the reversible ASCII embedding form."""

    if not isinstance(encoded_public_key, str):
        raise TypeError("encoded_public_key must be a string")

    text = encoded_public_key.strip()
    if not text:
        raise ValueError("encoded_public_key must not be empty")

    try:
        decoded = bytes.fromhex(text)
    except ValueError as exc:
        raise ValueError("encoded public key is not valid hexadecimal") from exc

    if len(decoded) != 32:
        raise ValueError("decoded public key length is not 32 bytes")

    return decoded


def derive_public_key_positions(
    public_key: bytes | bytearray | memoryview,
    *,
    number_of_positions: int | None = None,
    offset_do: int = 32,
    max_story_length: int = 20000,
    min_gap: int = 1,
    bit_chunk_size: int = 5,
) -> list[int]:
    """Generate deterministic SHAKE-128 positions for the embedded public key payload."""

    key_material = b"llm-shield/ecdhe-carrier/" + _normalize_public_key(public_key)
    encoded = public_key_to_embeddable_text(public_key)
    mapped = CharacterMap().encode(encoded)
    count = len(mapped) if number_of_positions is None else number_of_positions
    return generate_positions(
        key_material=key_material,
        number_of_positions=count,
        offset_do=offset_do,
        max_story_length=max_story_length,
        min_gap=min_gap,
        bit_chunk_size=bit_chunk_size,
    )


def embed_public_key_into_carrier(
    public_key: bytes | bytearray | memoryview,
    topic: str,
    *,
    initial_story: str = "",
    llm_generator: LLMGenerator | None = None,
    temperature: float = 0.7,
    top_k: int = 40,
    max_new_tokens: int = 32,
    max_attempts: int = 15000,
    max_retries: int = 3,
) -> tuple[str, list[int], str]:
    """Embed the public key representation into a natural carrier text."""

    key_bytes = _normalize_public_key(public_key)
    encoded = public_key_to_embeddable_text(key_bytes)
    mapped = CharacterMap().encode(encoded)
    positions = derive_public_key_positions(key_bytes, number_of_positions=len(mapped))
    embedder = EmbedderLLM(
        llm_generator=llm_generator if llm_generator is not None else LLMGenerator(),
        character_map=CharacterMap(),
    )
    result = embedder.embed(
        topic=topic,
        characters=mapped,
        positions=positions,
        initial_story=initial_story,
        temperature=temperature,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        max_attempts=max_attempts,
        max_retries=max_retries,
    )
    return result.story, positions, encoded


def extract_public_key_from_cover_text(
    cover_text: str,
    positions: Sequence[int],
) -> bytes:
    """Recover the original raw public key from a carrier text and recorded positions."""

    if not isinstance(cover_text, str):
        raise TypeError("cover_text must be a string")
    if not cover_text:
        raise ValueError("cover_text must not be empty")

    extracted = Extractor.extract(cover_text, positions)
    if not extracted:
        raise ValueError("No public-key payload was extracted from the carrier")

    try:
        encoded = CharacterMap().decode(extracted)
    except ValueError as exc:
        raise ValueError("Carrier extraction does not decode to a valid public-key payload") from exc

    try:
        public_key = decode_public_key_from_embedding(encoded)
    except ValueError as exc:
        raise ValueError("Recovered payload is not a valid X25519 public key") from exc

    return public_key


def _run_demo_flow(topic: str = "a quiet city street at dusk") -> None:
    print("[1] Alice generates X25519 key pair")
    alice_private, alice_public = generate_key_pair()
    bob_private, bob_public = generate_key_pair()
    print("Alice public key:", alice_public.hex())
    print("Bob public key   :", bob_public.hex())

    print("\n[2] Alice prepares public key for embedding")
    encoded_public_key = public_key_to_embeddable_text(alice_public)
    mapped_public_key = CharacterMap().encode(encoded_public_key)
    positions = derive_public_key_positions(alice_public, number_of_positions=len(mapped_public_key))
    print("Encoded public key:", encoded_public_key)
    print("Mapped payload length:", len(mapped_public_key))
    print("SHAKE-128 positions count:", len(positions))
    print("Sample positions:", positions[:10])

    print("\n[3] EmbedderLLM generates the carrier text")
    carrier_text, carrier_positions, _ = embed_public_key_into_carrier(
        alice_public,
        topic,
        initial_story="A quiet city street glows softly at dusk as a conversation begins.",
        temperature=0.7,
        top_k=40,
        max_new_tokens=32,
        max_attempts=15000,
        max_retries=3,
    )
    print("Carrier text:")
    print(carrier_text)

    print("\n[4] Bob receives the carrier text")
    print("Bob extracts hidden public-key data")
    recovered_public_key = extract_public_key_from_cover_text(carrier_text, carrier_positions)
    print("Recovered public key:", recovered_public_key.hex())
    print("Original public key :", alice_public.hex())
    print("Public key verification:", "PASS" if recovered_public_key == alice_public else "FAIL")
    if recovered_public_key != alice_public:
        raise RuntimeError("Recovered public key does not match Alice's original public key")

    print("\n[5] X25519 ECDHE")
    salt = b"llm-shield-carrier-demo"
    context = b"carrier-demo-session"
    alice_send, _ = derive_session_keys(alice_private, bob_public, salt, context)
    _, bob_receive = derive_session_keys(bob_private, recovered_public_key, salt, context)
    print("Alice ECDHE key:", alice_send.hex())
    print("Bob ECDHE key  :", bob_receive.hex())
    print("Shared secret match:", "PASS" if alice_send == bob_receive else "FAIL")
    if alice_send != bob_receive:
        raise RuntimeError("ECDHE session key mismatch")

    print("\n[6] AES-256-GCM secure communication")
    secret_message = b"hello from Alice through the carrier"
    packet = aead_encrypt(secret_message, alice_send, context)
    recovered_message = aead_decrypt(
        packet["ciphertext"],
        packet["tag"],
        packet["nonce"],
        bob_receive,
        context,
    )
    print("Original message:", secret_message.decode("utf-8"))
    print("Recovered message:", recovered_message.decode("utf-8"))
    print("AES-GCM match:", "PASS" if recovered_message == secret_message else "FAIL")
    if recovered_message != secret_message:
        raise RuntimeError("AES-GCM decryption did not recover the plaintext")

    print("\nINTEGRATED ECDHE + carrier demo: PASS")


def main() -> None:
    _run_demo_flow()


if __name__ == "__main__":
    main()
