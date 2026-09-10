"""Educational, pure-Python AES-256-GCM implementation."""

from __future__ import annotations

import hmac
import os
from typing import Any


class AEADError(ValueError):
    """Raised when authenticated decryption fails."""


_BLOCK_SIZE = 16
_TAG_SIZE = 16
_R = 0xE1000000000000000000000000000000


def _to_bytes(value: Any, name: str, *, allow_none: bool = False) -> bytes | None:
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
    key_bytes = _to_bytes(key, "key")
    if len(key_bytes) != 32:
        raise ValueError("AES-256 key must be exactly 32 bytes")
    return key_bytes


def _normalize_nonce(nonce: bytes | bytearray | memoryview) -> bytes:
    nonce_bytes = _to_bytes(nonce, "nonce")
    if len(nonce_bytes) != 12:
        raise ValueError("AES-GCM nonce must be exactly 12 bytes")
    return nonce_bytes


def _gf_mul(a: int, b: int) -> int:
    result = 0
    for _ in range(8):
        if b & 1:
            result ^= a
        a = ((a << 1) ^ 0x11B) if a & 0x80 else a << 1
        b >>= 1
    return result & 0xFF


def _gf_pow(value: int, exponent: int) -> int:
    result = 1
    while exponent:
        if exponent & 1:
            result = _gf_mul(result, value)
        value = _gf_mul(value, value)
        exponent >>= 1
    return result


def _rotl8(value: int, amount: int) -> int:
    return ((value << amount) | (value >> (8 - amount))) & 0xFF


def _sbox(value: int) -> int:
    inverse = 0 if value == 0 else _gf_pow(value, 254)
    return (inverse ^ _rotl8(inverse, 1) ^ _rotl8(inverse, 2) ^ _rotl8(inverse, 3) ^ _rotl8(inverse, 4) ^ 0x63)


def _rcon(index: int) -> int:
    value = 1
    for _ in range(index - 1):
        value = _gf_mul(value, 2)
    return value


def _expand_key(key: bytes) -> list[bytes]:
    words = [bytearray(key[index:index + 4]) for index in range(0, 32, 4)]
    for index in range(8, 60):
        word = bytearray(words[index - 1])
        if index % 8 == 0:
            word = bytearray(_sbox(byte) for byte in word[1:] + word[:1])
            word[0] ^= _rcon(index // 8)
        elif index % 8 == 4:
            word = bytearray(_sbox(byte) for byte in word)
        words.append(bytearray(a ^ b for a, b in zip(words[index - 8], word)))
    return [bytes(b"".join(words[index:index + 4])) for index in range(0, 60, 4)]


def _aes_encrypt_block(block: bytes, round_keys: list[bytes]) -> bytes:
    state = bytearray(a ^ b for a, b in zip(block, round_keys[0]))
    for round_key in round_keys[1:-1]:
        state = bytearray(_sbox(byte) for byte in state)
        state = bytearray(state[4 * ((column + row) % 4) + row] for column in range(4) for row in range(4))
        for column in range(4):
            offset = 4 * column
            a0, a1, a2, a3 = state[offset:offset + 4]
            state[offset:offset + 4] = bytes((
                _gf_mul(a0, 2) ^ _gf_mul(a1, 3) ^ a2 ^ a3,
                a0 ^ _gf_mul(a1, 2) ^ _gf_mul(a2, 3) ^ a3,
                a0 ^ a1 ^ _gf_mul(a2, 2) ^ _gf_mul(a3, 3),
                _gf_mul(a0, 3) ^ a1 ^ a2 ^ _gf_mul(a3, 2),
            ))
        state = bytearray(a ^ b for a, b in zip(state, round_key))
    state = bytearray(_sbox(byte) for byte in state)
    state = bytearray(state[4 * ((column + row) % 4) + row] for column in range(4) for row in range(4))
    return bytes(a ^ b for a, b in zip(state, round_keys[-1]))


def _inc32(counter: bytes) -> bytes:
    value = (int.from_bytes(counter[12:], "big") + 1) & 0xFFFFFFFF
    return counter[:12] + value.to_bytes(4, "big")


def _ctr_crypt(data: bytes, initial_counter: bytes, round_keys: list[bytes]) -> bytes:
    output = bytearray()
    counter = initial_counter
    for offset in range(0, len(data), _BLOCK_SIZE):
        counter = _inc32(counter)
        stream = _aes_encrypt_block(counter, round_keys)
        chunk = data[offset:offset + _BLOCK_SIZE]
        output.extend(byte ^ stream[index] for index, byte in enumerate(chunk))
    return bytes(output)


def _gcm_multiply(left: int, right: int) -> int:
    result = 0
    for bit in range(128):
        if (left >> (127 - bit)) & 1:
            result ^= right
        right = (right >> 1) ^ _R if right & 1 else right >> 1
    return result


def _ghash(data: bytes, hash_subkey: int) -> int:
    accumulator = 0
    for offset in range(0, len(data), _BLOCK_SIZE):
        block = data[offset:offset + _BLOCK_SIZE].ljust(_BLOCK_SIZE, b"\0")
        accumulator = _gcm_multiply(accumulator ^ int.from_bytes(block, "big"), hash_subkey)
    return accumulator


def _authentication_tag(aad: bytes, ciphertext: bytes, hash_subkey: int, j0: bytes, round_keys: list[bytes]) -> bytes:
    lengths = (len(aad) * 8).to_bytes(8, "big") + (len(ciphertext) * 8).to_bytes(8, "big")
    padded_aad = aad + b"\0" * ((-len(aad)) % 16)
    padded_ciphertext = ciphertext + b"\0" * ((-len(ciphertext)) % 16)
    hashed = _ghash(padded_aad + padded_ciphertext + lengths, hash_subkey)
    tag = int.from_bytes(_aes_encrypt_block(j0, round_keys), "big") ^ hashed
    return tag.to_bytes(_TAG_SIZE, "big")


def _j0(nonce: bytes) -> bytes:
    return nonce + b"\0\0\0\1"


def encrypt(plaintext: bytes | bytearray | memoryview, dk1: bytes | bytearray | memoryview, associated_data: bytes | bytearray | memoryview | None = None) -> dict[str, bytes]:
    """Encrypt plaintext and return nonce, ciphertext, tag, and Enc."""
    plaintext_bytes = _to_bytes(plaintext, "plaintext")
    key = _normalize_key(dk1)
    aad = _to_bytes(associated_data, "associated_data", allow_none=True) or b""
    nonce = os.urandom(12)
    round_keys = _expand_key(key)
    j0 = _j0(nonce)
    ciphertext = _ctr_crypt(plaintext_bytes, j0, round_keys)
    hash_subkey = int.from_bytes(_aes_encrypt_block(b"\0" * 16, round_keys), "big")
    tag = _authentication_tag(aad, ciphertext, hash_subkey, j0, round_keys)
    return {"nonce": nonce, "ciphertext": ciphertext, "tag": tag, "enc": tag + ciphertext}


def decrypt(ciphertext: bytes | bytearray | memoryview, tag: bytes | bytearray | memoryview, nonce: bytes | bytearray | memoryview, dk1: bytes | bytearray | memoryview, associated_data: bytes | bytearray | memoryview | None = None) -> bytes:
    """Authenticate and decrypt ciphertext."""
    ciphertext_bytes = _to_bytes(ciphertext, "ciphertext")
    tag_bytes = _to_bytes(tag, "tag")
    nonce_bytes = _normalize_nonce(nonce)
    key = _normalize_key(dk1)
    aad = _to_bytes(associated_data, "associated_data", allow_none=True) or b""
    if len(tag_bytes) != _TAG_SIZE:
        raise ValueError("authentication tag must be exactly 16 bytes")
    round_keys = _expand_key(key)
    j0 = _j0(nonce_bytes)
    hash_subkey = int.from_bytes(_aes_encrypt_block(b"\0" * 16, round_keys), "big")
    expected_tag = _authentication_tag(aad, ciphertext_bytes, hash_subkey, j0, round_keys)
    if not hmac.compare_digest(tag_bytes, expected_tag):
        raise AEADError("authentication failed: ciphertext or tag mismatch")
    return _ctr_crypt(ciphertext_bytes, j0, round_keys)


__all__ = ["AEADError", "decrypt", "encrypt"]
