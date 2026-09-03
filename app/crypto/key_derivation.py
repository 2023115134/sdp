"""PBKDF2-based key derivation for the cryptography phase.

This module intentionally isolates password-to-key derivation from the
existing Phase 1 SHAKE128 and LLM pipeline. It provides two independent
256-bit keys, dk1 and dk2, from a passphrase and a per-message salt.
"""

from __future__ import annotations

import hashlib
import os
from typing import Union

from app.config import DEFAULT_PBKDF2_CONFIG, get_pbkdf2_iterations

PasswordLike = Union[str, bytes, bytearray]
SaltLike = Union[str, bytes, bytearray]


def _normalize_password(password: PasswordLike) -> bytes:
    """Normalize a user password to bytes.

    Empty passwords are rejected because PBKDF2 with an empty password would
    silently accept invalid input and create ambiguous semantics in the wider
    protocol.
    """

    if password is None:
        raise TypeError("password must be str, bytes, or bytearray")

    if isinstance(password, str):
        if not password:
            raise ValueError("password must not be empty")
        return password.encode("utf-8")

    if isinstance(password, (bytes, bytearray)):
        if not password:
            raise ValueError("password must not be empty")
        return bytes(password)

    raise TypeError("password must be str, bytes, or bytearray")


def _normalize_salt(salt: SaltLike) -> bytes:
    """Normalize a salt to bytes.

    A random salt is required for the protocol. A zero-length salt is rejected
    because it does not provide meaningful domain separation.
    """

    if salt is None:
        raise TypeError("salt must be str, bytes, or bytearray")

    if isinstance(salt, str):
        if not salt:
            raise ValueError("salt must not be empty")
        return salt.encode("utf-8")

    if isinstance(salt, (bytes, bytearray)):
        if not salt:
            raise ValueError("salt must not be empty")
        return bytes(salt)

    raise TypeError("salt must be str, bytes, or bytearray")


def generate_salt(length: int = DEFAULT_PBKDF2_CONFIG.salt_length) -> bytes:
    """Generate a cryptographically secure random salt.

    The caller should store the salt alongside the derived keys so the same
    password can be re-derived for validation or future cryptographic use.
    """

    if length <= 0:
        raise ValueError("salt length must be > 0")

    return os.urandom(length)


def derive_keys(password: PasswordLike, salt: SaltLike) -> tuple[bytes, bytes]:
    """Derive two independent 256-bit keys using PBKDF2-HMAC-SHA256.

    The implementation derives a single PBKDF2 stream of length 64 bytes and
    splits it into two 32-byte keys:

        dk1 = derived_key[:32]
        dk2 = derived_key[32:64]

    This is simpler and more robust than separate PBKDF2 calls because it keeps
    the password and salt interaction within the same keyed expansion while
    creating two independent keys with a single deterministic pool of output.

    Args:
        password: Password or passphrase, as str/bytes.
        salt: Salt bytes, as str/bytes.

    Returns:
        A tuple (dk1, dk2), each exactly 32 bytes long.
    """

    normalized_password = _normalize_password(password)
    normalized_salt = _normalize_salt(salt)

    iterations = get_pbkdf2_iterations()
    if iterations <= 0:
        raise ValueError("PBKDF2 iterations must be > 0")

    derived_key = hashlib.pbkdf2_hmac(
        hash_name=DEFAULT_PBKDF2_CONFIG.hash_name,
        password=normalized_password,
        salt=normalized_salt,
        iterations=iterations,
        dklen=DEFAULT_PBKDF2_CONFIG.key_length,
    )

    dk1 = derived_key[:32]
    dk2 = derived_key[32:64]

    if len(dk1) != 32 or len(dk2) != 32:
        raise ValueError("PBKDF2 output length mismatch")

    return dk1, dk2


__all__ = ["derive_keys", "generate_salt"]
