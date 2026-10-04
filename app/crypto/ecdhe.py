"""Ephemeral X25519 key exchange with HKDF-derived AES-GCM encryption.

This module supports two ECDHE flows:

1. Existing packet flow:
   X25519 -> HKDF-SHA256 -> AES-256-GCM

2. Secure LLM-Shield project flow:
   X25519 -> raw shared secret -> PBKDF2 -> DK1/DK2
   -> AES-256-GCM + SHAKE-128 target positions
"""

from __future__ import annotations

import os
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .aead import decrypt as aead_decrypt
from .aead import encrypt as aead_encrypt
from .key_derivation import derive_keys


class ECDHEError(ValueError):
    """Raised when an ECDHE packet or key is invalid."""


_PROTOCOL_LABEL = b"llm-shield/ecdhe/x25519/aes-256-gcm/v1"

# X25519 private/public keys are both 32 bytes.
_KEY_SIZE = 32

# Salt used by the existing HKDF packet flow.
_SALT_SIZE = 16


def generate_key_pair() -> tuple[bytes, bytes]:
    """Generate a fresh X25519 private/public key pair.

    Returns:
        tuple:
            private_key: 32-byte raw private key
            public_key: 32-byte raw public key
    """

    private_key = X25519PrivateKey.generate()

    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )

    private_bytes = private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )

    return private_bytes, public_key


def _normalize_key(
    value: bytes | bytearray | memoryview,
    name: str,
) -> bytes:
    """Validate and normalize an X25519 key."""

    if isinstance(value, memoryview):
        value = value.tobytes()

    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"{name} must be bytes-like")

    value = bytes(value)

    if len(value) != _KEY_SIZE:
        raise ECDHEError(
            f"{name} must be exactly {_KEY_SIZE} bytes"
        )

    return value


def derive_shared_secret(
    private_key: bytes | bytearray | memoryview,
    peer_public_key: bytes | bytearray | memoryview,
) -> bytes:
    """Return the raw 32-byte X25519 shared secret."""

    private_bytes = _normalize_key(
        private_key,
        "private_key",
    )

    peer_public = _normalize_key(
        peer_public_key,
        "peer_public_key",
    )

    try:
        local_private = X25519PrivateKey.from_private_bytes(private_bytes)
        local_public = local_private.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )

        if local_public == peer_public:
            raise ECDHEError(
                "peer public key must differ from the local public key"
            )

        peer_key = X25519PublicKey.from_public_bytes(peer_public)
        shared_secret = local_private.exchange(peer_key)
    except ECDHEError:
        raise
    except (TypeError, ValueError) as error:
        raise ECDHEError("invalid X25519 key") from error

    if len(shared_secret) != 32:
        raise ECDHEError(
            "X25519 shared secret must be exactly 32 bytes"
        )

    return shared_secret


def derive_ecdhe_keys(
    shared_secret: bytes | bytearray | memoryview,
    salt: bytes | bytearray | memoryview,
) -> tuple[bytes, bytes]:
    """Derive the AES-GCM key (DK1) and position key (DK2) from the raw ECDHE secret."""

    if isinstance(shared_secret, memoryview):
        shared_secret = shared_secret.tobytes()
    if not isinstance(shared_secret, (bytes, bytearray)):
        raise TypeError("shared_secret must be bytes-like")
    shared_secret_bytes = bytes(shared_secret)
    if len(shared_secret_bytes) != _KEY_SIZE:
        raise ECDHEError("shared_secret must be exactly 32 bytes")

    if isinstance(salt, memoryview):
        salt = salt.tobytes()
    if not isinstance(salt, (bytes, bytearray, str)):
        raise TypeError("salt must be bytes-like or str")
    if isinstance(salt, str):
        salt_bytes = salt.encode("utf-8")
    else:
        salt_bytes = bytes(salt)
    if not salt_bytes:
        raise ECDHEError("salt must not be empty")

    return derive_keys(shared_secret_bytes, salt_bytes)


def _derive_key(
    private_key: bytes,
    peer_public_key: bytes,
    salt: bytes,
    associated_data: bytes,
) -> bytes:
    """Existing HKDF-based key derivation.

    This function is retained for backward compatibility with the
    existing ECDHE packet encryption/decryption flow.
    """

    try:
        shared_secret = X25519PrivateKey.from_private_bytes(
            private_key
        ).exchange(
            X25519PublicKey.from_public_bytes(
                peer_public_key
            )
        )
    except (TypeError, ValueError) as error:
        raise ECDHEError(
            "invalid X25519 key"
        ) from error

    context = (
        _PROTOCOL_LABEL
        + len(associated_data).to_bytes(8, "big")
        + associated_data
    )

    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=context,
    ).derive(shared_secret)


def derive_session_keys(
    private_key: bytes | bytearray | memoryview,
    peer_public_key: bytes | bytearray | memoryview,
    salt: bytes | bytearray | memoryview,
    associated_data: bytes | bytearray | memoryview | None = None,
) -> tuple[bytes, bytes]:
    """Derive matching directional keys using X25519 + HKDF.

    This is the existing duplex ECDHE path and is kept unchanged
    for compatibility.
    """

    private_bytes = _normalize_key(
        private_key,
        "private_key",
    )

    peer_public = _normalize_key(
        peer_public_key,
        "peer_public_key",
    )

    if isinstance(salt, memoryview):
        salt = salt.tobytes()

    if not isinstance(salt, (bytes, bytearray)):
        raise ECDHEError(
            "salt must be exactly 16 bytes"
        )

    if len(salt) != _SALT_SIZE:
        raise ECDHEError(
            "salt must be exactly 16 bytes"
        )

    salt_bytes = bytes(salt)

    if associated_data is None:
        aad = b""
    elif isinstance(associated_data, memoryview):
        aad = associated_data.tobytes()
    elif isinstance(associated_data, (bytes, bytearray)):
        aad = bytes(associated_data)
    else:
        raise TypeError(
            "associated_data must be bytes-like or None"
        )

    try:
        local_private = X25519PrivateKey.from_private_bytes(
            private_bytes
        )

        local_public = local_private.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )

        shared_secret = local_private.exchange(
            X25519PublicKey.from_public_bytes(
                peer_public
            )
        )
    except (TypeError, ValueError) as error:
        raise ECDHEError(
            "invalid X25519 key"
        ) from error

    if local_public == peer_public:
        raise ECDHEError(
            "peer public key must differ from the local public key"
        )

    first_public, second_public = sorted(
        (local_public, peer_public)
    )

    context = (
        _PROTOCOL_LABEL
        + b"/duplex"
        + len(aad).to_bytes(8, "big")
        + aad
        + first_public
        + second_public
    )

    key_material = HKDF(
        algorithm=hashes.SHA256(),
        length=64,
        salt=salt_bytes,
        info=context,
    ).derive(shared_secret)

    first_to_second = key_material[:32]
    second_to_first = key_material[32:]

    if local_public == first_public:
        return first_to_second, second_to_first

    return second_to_first, first_to_second


def _packet_associated_data(
    ephemeral_public_key: bytes,
    associated_data: bytes,
) -> bytes:
    """Build authenticated packet metadata."""

    return (
        _PROTOCOL_LABEL
        + ephemeral_public_key
        + associated_data
    )


def encrypt_for_peer(
    plaintext: bytes | bytearray | memoryview,
    recipient_public_key: bytes | bytearray | memoryview,
    associated_data: bytes | bytearray | memoryview | None = None,
) -> dict[str, bytes]:
    """Encrypt a message for a recipient using a fresh ephemeral key pair.

    Existing flow:

        Ephemeral X25519
              ↓
        Shared Secret
              ↓
        HKDF-SHA256
              ↓
        AES-256-GCM
    """

    recipient_public = _normalize_key(
        recipient_public_key,
        "recipient_public_key",
    )

    if associated_data is None:
        aad = b""
    elif isinstance(associated_data, memoryview):
        aad = associated_data.tobytes()
    elif isinstance(associated_data, (bytes, bytearray)):
        aad = bytes(associated_data)
    else:
        raise TypeError(
            "associated_data must be bytes-like or None"
        )

    ephemeral_private, ephemeral_public = generate_key_pair()
    salt = os.urandom(_SALT_SIZE)

    key = _derive_key(
        ephemeral_private,
        recipient_public,
        salt,
        aad,
    )

    encrypted = aead_encrypt(
        plaintext,
        key,
        _packet_associated_data(
            ephemeral_public,
            aad,
        ),
    )

    return {
        "version": b"1",
        "ephemeral_public_key": ephemeral_public,
        "salt": salt,
        "nonce": encrypted["nonce"],
        "ciphertext": encrypted["ciphertext"],
        "tag": encrypted["tag"],
    }


def decrypt_from_peer(
    packet: dict[str, Any],
    recipient_private_key: bytes | bytearray | memoryview,
    associated_data: bytes | bytearray | memoryview | None = None,
) -> bytes:
    """Decrypt a packet produced by encrypt_for_peer()."""

    required_fields = (
        "version",
        "ephemeral_public_key",
        "salt",
        "nonce",
        "ciphertext",
        "tag",
    )

    if (
        not isinstance(packet, dict)
        or any(
            field not in packet
            for field in required_fields
        )
    ):
        raise ECDHEError(
            "packet is missing a required field"
        )

    if packet["version"] != b"1":
        raise ECDHEError(
            "unsupported ECDHE packet version"
        )

    private_key = _normalize_key(
        recipient_private_key,
        "recipient_private_key",
    )

    ephemeral_public = _normalize_key(
        packet["ephemeral_public_key"],
        "ephemeral_public_key",
    )

    salt = packet["salt"]

    if not isinstance(salt, bytes):
        raise ECDHEError(
            "salt must be exactly 16 bytes"
        )

    if len(salt) != _SALT_SIZE:
        raise ECDHEError(
            "salt must be exactly 16 bytes"
        )

    if associated_data is None:
        aad = b""
    elif isinstance(associated_data, memoryview):
        aad = associated_data.tobytes()
    elif isinstance(associated_data, (bytes, bytearray)):
        aad = bytes(associated_data)
    else:
        raise TypeError(
            "associated_data must be bytes-like or None"
        )

    key = _derive_key(
        private_key,
        ephemeral_public,
        salt,
        aad,
    )

    return aead_decrypt(
        packet["ciphertext"],
        packet["tag"],
        packet["nonce"],
        key,
        _packet_associated_data(
            ephemeral_public,
            aad,
        ),
    )


__all__ = [
    "ECDHEError",
    "decrypt_from_peer",
    "derive_ecdhe_keys",
    "derive_shared_secret",
    "derive_session_keys",
    "encrypt_for_peer",
    "generate_key_pair",
]
