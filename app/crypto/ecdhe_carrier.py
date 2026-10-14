"""Transport X25519 public-key segments in LLM-generated carrier text."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence
from typing import TYPE_CHECKING

from app.crypto.key_derivation import derive_keys
from app.crypto.mapping import CharacterMap
from app.crypto.position_generator import generate_positions
from app.extraction.extractor import Extractor

if TYPE_CHECKING:
    from app.llm.embedder import EmbedderLLM

PUBLIC_KEY_SEGMENT_COUNT = 5
PUBLIC_KEY_EXCHANGE_MESSAGE_COUNT = PUBLIC_KEY_SEGMENT_COUNT * 2
_PUBLIC_KEY_SEGMENT_LENGTHS = (7, 7, 6, 6, 6)
_MAX_CARRIER_LENGTH = 100_000


def split_public_key(public_key: bytes) -> list[bytes]:
    """Split a raw X25519 public key into five ordered carrier payloads."""

    if not isinstance(public_key, bytes) or len(public_key) != 32:
        raise ValueError("public key must be exactly 32 bytes")
    segments: list[bytes] = []
    offset = 0
    for segment_length in _PUBLIC_KEY_SEGMENT_LENGTHS:
        segments.append(public_key[offset : offset + segment_length])
        offset += segment_length
    return segments


def _expected_segment_length(segment_index: int) -> int:
    if type(segment_index) is not int or not 0 <= segment_index < PUBLIC_KEY_SEGMENT_COUNT:
        raise ValueError("public-key segment index must be between 0 and 4")
    return _PUBLIC_KEY_SEGMENT_LENGTHS[segment_index]


def derive_carrier_position_keys(
    bootstrap_secret: str,
    session_id: bytes,
) -> tuple[bytes, bytes]:
    """Derive the matching directional keys used to place key-carrier text."""

    if not isinstance(bootstrap_secret, str):
        raise TypeError("bootstrap secret must be a string")
    if not isinstance(session_id, bytes) or len(session_id) != 16:
        raise ValueError("ECDHE carrier session ID must be exactly 16 bytes")
    return derive_keys(
        bootstrap_secret,
        session_id,
        allow_empty_password=True,
    )


def _position_key(
    directional_key: bytes,
    role: str,
    segment_index: int,
) -> bytes:
    if not isinstance(directional_key, bytes) or len(directional_key) != 32:
        raise ValueError("directional position key must be exactly 32 bytes")
    if role not in {"server_a", "server_b"}:
        raise ValueError("role must be 'server_a' or 'server_b'")
    _expected_segment_length(segment_index)
    transcript = (
        b"llm-shield/ecdhe-public-key-carrier/v2/"
        + role.encode("ascii")
        + b"/"
        + segment_index.to_bytes(1, "big")
    )
    return hmac.new(directional_key, transcript, hashlib.sha256).digest()


def embed_public_key_segment(
    segment: bytes,
    *,
    role: str,
    segment_index: int,
    topic: str,
    position_key: bytes,
    embedder: EmbedderLLM | None = None,
) -> dict[str, object]:
    """Map and embed one public-key segment, returning its carrier frame fields."""

    expected_length = _expected_segment_length(segment_index)
    if not isinstance(segment, bytes) or len(segment) != expected_length:
        raise ValueError(
            f"public-key segment {segment_index} must contain {expected_length} bytes"
        )
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("topic must be a non-empty string")

    mapped = CharacterMap().encode(segment.hex())
    positions = generate_positions(
        key_material=_position_key(position_key, role, segment_index),
        number_of_positions=len(mapped),
        offset_do=32,
        max_story_length=_MAX_CARRIER_LENGTH,
        min_gap=1,
    )

    from app.llm.embedder import EmbedderLLM

    if embedder is None:
        from app.llm.generator import LLMGenerator

        embedder = EmbedderLLM(
            llm_generator=LLMGenerator(),
            character_map=CharacterMap(),
        )
    initial_story = topic.strip()[: max(1, positions[0] - 1)].rstrip()
    if not initial_story:
        initial_story = "A quiet story"
    while len(initial_story) >= positions[0]:
        initial_story = initial_story[:-1].rstrip() or "A quiet story"

    result = embedder.embed(
        topic=topic,
        characters=mapped,
        positions=positions,
        initial_story=initial_story,
        temperature=0.7,
        top_k=20,
        max_new_tokens=32,
        max_attempts=15000,
        max_retries=3,
        allow_adaptive_positions=False,
        verbose=False,
    )
    actual_positions = result.positions
    if (
        len(actual_positions) != len(mapped)
        or any(left >= right for left, right in zip(actual_positions, actual_positions[1:]))
        or not EmbedderLLM.verify_embedding(result.story, mapped, actual_positions)
    ):
        raise RuntimeError("LLM public-key segment embedding verification failed")

    return {
        "type": "public_key_carrier",
        "role": role,
        "segment_index": segment_index,
        "cover_text": result.story,
    }


def extract_public_key_segment(
    message: dict[str, object],
    *,
    expected_role: str,
    expected_segment_index: int,
    position_key: bytes,
) -> bytes:
    """Extract and validate one mapped public-key segment from a carrier frame."""

    _expected_segment_length(expected_segment_index)
    if (
        message.get("type") != "public_key_carrier"
        or message.get("role") != expected_role
        or message.get("segment_index") != expected_segment_index
    ):
        raise ValueError("received public-key carrier is out of handshake order")

    cover_text = message.get("cover_text")
    expected_mapped_length = 4 * _expected_segment_length(expected_segment_index)
    if not isinstance(cover_text, str) or not cover_text:
        raise ValueError("public-key carrier text must be non-empty")
    positions = generate_positions(
        key_material=_position_key(
            position_key,
            expected_role,
            expected_segment_index,
        ),
        number_of_positions=expected_mapped_length,
        offset_do=32,
        max_story_length=_MAX_CARRIER_LENGTH,
        min_gap=1,
    )
    if positions[-1] >= len(cover_text):
        raise ValueError("public-key carrier text is too short for extracted positions")

    mapped = Extractor.extract(cover_text, positions)
    encoded_segment = CharacterMap().decode(mapped)
    try:
        segment = bytes.fromhex(encoded_segment)
    except ValueError as error:
        raise ValueError("extracted public-key segment is not valid hexadecimal") from error
    if len(segment) != _expected_segment_length(expected_segment_index):
        raise ValueError("extracted public-key segment has an invalid length")
    return segment


def assemble_public_key(segments: Sequence[bytes]) -> bytes:
    """Join five extracted segments and validate the resulting X25519 key size."""

    if len(segments) != PUBLIC_KEY_SEGMENT_COUNT:
        raise ValueError("an X25519 public key requires exactly five segments")
    public_key = b"".join(segments)
    if len(public_key) != 32:
        raise ValueError("assembled X25519 public key must be exactly 32 bytes")
    return public_key


__all__ = [
    "PUBLIC_KEY_EXCHANGE_MESSAGE_COUNT",
    "PUBLIC_KEY_SEGMENT_COUNT",
    "assemble_public_key",
    "derive_carrier_position_keys",
    "embed_public_key_segment",
    "extract_public_key_segment",
    "split_public_key",
]
