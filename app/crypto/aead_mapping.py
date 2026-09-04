"""Adapters for converting AEAD output into the project's h4 representation.

This module provides a boundary adapter between the authenticated encrypted
representation and the reversible h4 alphabet.

The Phase-2 preprocessing is:

    Plaintext
        -> AES-256-GCM
        -> Enc = Tag || Ciphertext
        -> direct high/low nibble mapping
        -> C

The nonce remains separate and is not folded into the logical Enc value.
"""

from __future__ import annotations

from app.crypto.aead import decrypt
from app.crypto.mapping import CharacterMap


def aead_to_character_sequence(enc: bytes | bytearray | memoryview) -> str:
    """Convert AEAD Enc = Tag || Ciphertext into h4 characters."""

    if isinstance(enc, memoryview):
        enc_bytes = enc.tobytes()
    elif isinstance(enc, (bytes, bytearray)):
        enc_bytes = bytes(enc)
    else:
        raise TypeError("enc must be bytes-like")

    return CharacterMap().encode(enc_bytes)


def character_sequence_to_aead(mapped: str) -> bytes:
    """Reverse the h4 mapping back to the underlying AEAD Enc bytes."""

    if not isinstance(mapped, str):
        raise TypeError("mapped must be a string")

    if len(mapped) % 2 != 0:
        raise ValueError("Mapped h4 sequence must contain an even number of characters.")

    values = CharacterMap().to_values(mapped)
    return bytes(
        (high << 4) | low
        for high, low in zip(values[::2], values[1::2])
    )


def decrypt_aead_from_mapped(
    mapped: str,
    nonce: bytes | bytearray | memoryview,
    dk1: bytes | bytearray | memoryview,
) -> bytes:
    """Recover Enc from the mapped string and decrypt it using the original nonce."""

    enc = character_sequence_to_aead(mapped)
    if len(enc) < 16:
        raise ValueError("mapped AEAD payload is too short to contain a valid tag")

    tag = enc[:16]
    ciphertext = enc[16:]
    return decrypt(ciphertext=ciphertext, tag=tag, nonce=nonce, dk1=dk1)


__all__ = [
    "aead_to_character_sequence",
    "character_sequence_to_aead",
    "decrypt_aead_from_mapped",
]
