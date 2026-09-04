"""Derive cryptographic keys from a password and salt with PBKDF2."""

from __future__ import annotations

import hashlib
import os
from typing import Union

from app.config import DEFAULT_PBKDF2_CONFIG, get_pbkdf2_iterations

PasswordLike = Union[str, bytes, bytearray]
SaltLike = Union[str, bytes, bytearray]


def _normalize_password(password: PasswordLike) -> bytes:
    """Convert a non-empty password to bytes."""

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
    """Convert a non-empty salt to bytes."""

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
    """Generate a random salt of the configured length."""

    if length <= 0:
        raise ValueError("salt length must be > 0")

    return os.urandom(length)


def derive_keys(password: PasswordLike, salt: SaltLike) -> tuple[bytes, bytes]:
    """Derive 64 bytes with PBKDF2-HMAC-SHA256 and split them into two keys."""

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
