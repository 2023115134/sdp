"""Helpers for embedding and extracting raw X25519 public keys in cover text."""

from __future__ import annotations

from collections.abc import Sequence

from app.crypto.mapping import CharacterMap
from app.extraction.extractor import Extractor


def public_key_to_embeddable_text(public_key: bytes | bytearray | memoryview) -> str:
    """Convert a raw X25519 public key into a hex string suitable for embedding."""
    if isinstance(public_key, memoryview):
        public_key = public_key.tobytes()

    if not isinstance(public_key, (bytes, bytearray)):
        raise TypeError("public_key must be bytes-like")

    key = bytes(public_key)
    if len(key) != 32:
        raise ValueError("X25519 public keys must be exactly 32 bytes")

    return key.hex()


def decode_public_key_from_embedding(encoded: str) -> bytes:
    """Decode a hex-encoded public key that was embedded in a carrier text."""
    if not isinstance(encoded, str):
        raise TypeError("encoded public key must be a string")

    candidate = encoded.strip()
    if not candidate:
        raise ValueError("public key payload must not be empty")

    if len(candidate) % 2 != 0:
        raise ValueError("embedded public key must have an even number of hex characters")

    try:
        value = bytes.fromhex(candidate)
    except ValueError as exc:
        raise ValueError("embedded public key is not valid hexadecimal") from exc

    if len(value) != 32:
        raise ValueError("decoded public key must be exactly 32 bytes")

    return value


def extract_public_key_from_cover_text(cover_text: str, positions: Sequence[int]) -> bytes:
    """Read a mapped public key from a carrier and return the original 32-byte key."""
    if not isinstance(cover_text, str):
        raise TypeError("cover_text must be a string")

    if positions is None:
        raise ValueError("positions must not be None")

    extracted = Extractor.extract(cover_text, positions).upper()
    if not extracted:
        raise ValueError("no public key material was extracted from the cover text")

    try:
        decoded_hex = CharacterMap().decode(extracted)
    except ValueError as exc:
        raise ValueError("public-key carrier contains malformed h4-encoded data") from exc

    return decode_public_key_from_embedding(decoded_hex)


__all__ = [
    "decode_public_key_from_embedding",
    "extract_public_key_from_cover_text",
    "public_key_to_embeddable_text",
]
