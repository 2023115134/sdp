"""AES-256-GCM encryption for the protocol's AEAD layer."""

from __future__ import annotations

import os
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM 


class AEADError(ValueError):
    """Raised when authenticated decryption fails."""


def _to_bytes(value: Any, name: str, *, allow_none: bool = False) -> bytes | None:
    """Normalize bytes-like input and reject unsupported values."""

    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{name} must be bytes-like")

    if isinstance(value, memoryview):
        return value.tobytes()

    if isinstance(value, (bytes, bytearray)):
        return bytes(value)

    raise TypeError(f"{name} must be bytes-like")


def _normalize_key(key: bytes | bytearray | memoryview) -> bytes:
    """Normalize and validate an AES-256 key."""
    key_bytes = _to_bytes(key, "key")
    if len(key_bytes) != 32:
        raise ValueError("AES-256 key must be exactly 32 bytes")
    return key_bytes


def _normalize_nonce(nonce: bytes | bytearray | memoryview) -> bytes:
    """Normalize and validate an AES-GCM nonce."""
    nonce_bytes = _to_bytes(nonce, "nonce")
    if len(nonce_bytes) != 12:
        raise ValueError("AES-GCM nonce must be exactly 12 bytes")
    return nonce_bytes


def encrypt(
    plaintext: bytes | bytearray | memoryview,
    dk1: bytes | bytearray | memoryview,
    associated_data: bytes | bytearray | memoryview | None = None,
) -> dict[str, bytes]:
    """Encrypt plaintext and return its nonce, tag, ciphertext, and encoding."""

    plaintext_bytes = _to_bytes(plaintext, "plaintext")
    key = _normalize_key(dk1)
    aad = _to_bytes(associated_data, "associated_data", allow_none=True)

    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    sealed = aesgcm.encrypt(nonce, plaintext_bytes, aad)

    tag = sealed[-16:]
    ciphertext = sealed[:-16]

    return {
        "nonce": nonce,
        "ciphertext": ciphertext,
        "tag": tag,
        "enc": tag + ciphertext,
    }


def decrypt(
    ciphertext: bytes | bytearray | memoryview,
    tag: bytes | bytearray | memoryview,
    nonce: bytes | bytearray | memoryview,
    dk1: bytes | bytearray | memoryview,
    associated_data: bytes | bytearray | memoryview | None = None,
) -> bytes:
    """Authenticate and decrypt ciphertext using AES-256-GCM."""

    ciphertext_bytes = _to_bytes(ciphertext, "ciphertext")
    tag_bytes = _to_bytes(tag, "tag")
    nonce_bytes = _normalize_nonce(nonce)
    key = _normalize_key(dk1)
    aad = _to_bytes(associated_data, "associated_data", allow_none=True)

    if len(tag_bytes) != 16:
        raise ValueError("authentication tag must be exactly 16 bytes")

    aesgcm = AESGCM(key)
    sealed = ciphertext_bytes + tag_bytes

    try:
        return aesgcm.decrypt(nonce_bytes, sealed, aad)
    except InvalidTag as exc:
        raise AEADError("authentication failed: ciphertext or tag mismatch") from exc
    except ValueError as exc:
        raise AEADError(f"invalid AEAD input: {exc}") from exc


__all__ = ["AEADError", "decrypt", "encrypt"]
