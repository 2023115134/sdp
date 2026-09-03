"""Authenticated encryption using AES-256-GCM.

This module provides the Phase-2 AEAD layer independently of the PBKDF2 key
expansion and the existing Phase-1 SHAKE128 pipeline. The implementation uses
AES-256-GCM from the well-tested Python cryptography library because the paper
specifies generic AEAD semantics without locking to a particular primitive.

Concrete choice:
    AES-256-GCM

Properties:
    - key: 256 bits / 32 bytes (dk1 from PBKDF2)
    - nonce: 96 bits / 12 bytes, freshly generated per encryption
    - tag: 128 bits / 16 bytes
    - representation: Enc = Tag || Ciphertext
    - nonce is kept separate from the ciphertext because AES-GCM requires it for
      decryption and the protocol must preserve it explicitly.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from cryptography.exceptions import InvalidTag # type: ignore
from cryptography.hazmat.primitives.ciphers.aead import AESGCM # type: ignore


class AEADError(ValueError):
    """Raised when authenticated decryption fails or invalid AEAD inputs are supplied."""


def _to_bytes(value: Any, name: str, *, allow_none: bool = False) -> bytes | None:
    """Normalize bytes-like inputs while rejecting unsupported values."""

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


def _normalize_associated_data(associated_data: Any) -> bytes | None:
    """Normalize optional associated data."""

    return _to_bytes(associated_data, "associated_data", allow_none=True)


def _normalize_plaintext(plaintext: Any) -> bytes:
    """Normalize plaintext to bytes."""

    return _to_bytes(plaintext, "plaintext")


def encrypt(
    plaintext: bytes | bytearray | memoryview,
    dk1: bytes | bytearray | memoryview,
    associated_data: bytes | bytearray | memoryview | None = None,
) -> dict[str, bytes]:
    """Encrypt plaintext using AES-256-GCM and the provided key.

    Args:
        plaintext: Arbitrary plaintext bytes.
        dk1: AES key material, expected to be exactly 32 bytes from PBKDF2.
        associated_data: Optional authenticated but unencrypted metadata.

    Returns:
        A dictionary containing:
            - nonce: fresh 12-byte nonce
            - ciphertext: ciphertext without the authentication tag
            - tag: 16-byte authentication tag
            - enc: Tag || Ciphertext, matching the paper representation
    """

    plaintext_bytes = _normalize_plaintext(plaintext)
    key = _normalize_key(dk1)
    aad = _normalize_associated_data(associated_data)

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
    """Decrypt and authenticate ciphertext using AES-256-GCM.

    Authentication is verified before any plaintext is returned. Any tampering
    with ciphertext, tag, nonce, key, or associated data results in a failure.
    """

    ciphertext_bytes = _to_bytes(ciphertext, "ciphertext")
    tag_bytes = _to_bytes(tag, "tag")
    nonce_bytes = _normalize_nonce(nonce)
    key = _normalize_key(dk1)
    aad = _normalize_associated_data(associated_data)

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
