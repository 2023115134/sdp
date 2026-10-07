"""Length-prefixed JSON framing and shared helpers for the TCP ECDHE demo."""

from __future__ import annotations

import hashlib
import hmac
import json
import socket
import struct

from app.crypto.position_generator import generate_positions

MAX_FRAME_SIZE = 8 * 1024 * 1024
MAX_MAPPED_CHARACTERS = 8192
_PAYLOAD_AAD = b"llm-shield/tcp-ecdhe/aes-256-gcm/v1"


def send_json(connection: socket.socket, message: dict[str, object]) -> None:
    """Send one UTF-8 JSON object preceded by its four-byte network length."""

    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_FRAME_SIZE:
        raise ValueError(f"TCP frame exceeds {MAX_FRAME_SIZE} bytes")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("TCP peer closed the connection mid-frame")
        chunks.extend(chunk)
    return bytes(chunks)


def receive_json(connection: socket.socket) -> dict[str, object]:
    """Read one length-prefixed UTF-8 JSON object."""

    (frame_size,) = struct.unpack("!I", _receive_exact(connection, 4))
    if frame_size == 0 or frame_size > MAX_FRAME_SIZE:
        raise ValueError(f"invalid TCP frame size: {frame_size}")

    try:
        message = json.loads(_receive_exact(connection, frame_size))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("received TCP frame is not valid UTF-8 JSON") from error

    if not isinstance(message, dict):
        raise ValueError("TCP frame must contain a JSON object")
    return message


def decode_hex_field(
    message: dict[str, object],
    field: str,
    expected_size: int,
) -> bytes:
    """Decode a fixed-size hexadecimal field from a protocol message."""

    value = message.get(field)
    if not isinstance(value, str):
        raise ValueError(f"TCP message field {field!r} must be hexadecimal text")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"TCP message field {field!r} is not valid hexadecimal") from error
    if len(decoded) != expected_size:
        raise ValueError(
            f"TCP message field {field!r} must contain exactly {expected_size} bytes"
        )
    return decoded


def key_confirmation(
    shared_secret: bytes,
    role: str,
    server_a_public_key: bytes,
    server_b_public_key: bytes,
) -> str:
    """Create a role-separated HMAC proving possession of the X25519 secret."""

    if len(shared_secret) != 32:
        raise ValueError("shared secret must be exactly 32 bytes")
    if len(server_a_public_key) != 32 or len(server_b_public_key) != 32:
        raise ValueError("X25519 public keys must be exactly 32 bytes")
    if role not in {"server_a", "server_b"}:
        raise ValueError("role must be 'server_a' or 'server_b'")

    transcript = (
        b"llm-shield/tcp-ecdhe/key-confirmation/v1/"
        + role.encode("ascii")
        + server_a_public_key
        + server_b_public_key
    )
    return hmac.new(shared_secret, transcript, hashlib.sha256).hexdigest()


def payload_associated_data(mapped_length: int) -> bytes:
    """Bind the extracted payload length to its AES-GCM authentication tag."""

    if (
        not isinstance(mapped_length, int)
        or mapped_length <= 0
        or mapped_length % 2 != 0
        or mapped_length > MAX_MAPPED_CHARACTERS
    ):
        raise ValueError(
            "mapped payload length must be a positive even number no larger "
            f"than {MAX_MAPPED_CHARACTERS}"
        )
    return _PAYLOAD_AAD + mapped_length.to_bytes(4, "big")


def positions_for_payload(position_key: bytes, mapped_length: int) -> list[int]:
    """Generate shared SHAKE-128 positions for a mapped AES-GCM payload."""

    payload_associated_data(mapped_length)
    return generate_positions(
        key_material=position_key,
        number_of_positions=mapped_length,
        offset_do=32,
        max_story_length=max(100_000, 32 + mapped_length * 64),
        min_gap=1,
    )


def authenticate_positions(position_key: bytes, positions: list[int]) -> str:
    """Authenticate the final target sequence, including adaptive placements."""

    if len(position_key) != 32:
        raise ValueError("position key must be exactly 32 bytes")
    if not positions or any(type(position) is not int or position < 0 for position in positions):
        raise ValueError("positions must be a non-empty list of non-negative integers")
    if any(left >= right for left, right in zip(positions, positions[1:])):
        raise ValueError("positions must be strictly increasing")

    transcript = (
        b"llm-shield/tcp-ecdhe/adaptive-positions/v1"
        + len(positions).to_bytes(4, "big")
        + b"".join(position.to_bytes(8, "big") for position in positions)
    )
    return hmac.new(position_key, transcript, hashlib.sha256).hexdigest()


__all__ = [
    "MAX_MAPPED_CHARACTERS",
    "authenticate_positions",
    "decode_hex_field",
    "key_confirmation",
    "payload_associated_data",
    "positions_for_payload",
    "receive_json",
    "send_json",
]
