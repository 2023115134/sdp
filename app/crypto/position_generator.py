"""Generate deterministic embedding positions with SHAKE128."""

from __future__ import annotations

import hashlib
class PositionGenerator:
    """Generate deterministic positions subject to configured gaps."""

    def __init__(
        self,
        offset_do: int = 32,
        max_story_length: int = 100_000,
        min_gap: int = 20,
        bit_chunk_size: int = 5,
    ) -> None:
        if offset_do < 0:
            raise ValueError(
                "offset_do must be >= 0"
            )

        if max_story_length <= 0:
            raise ValueError(
                "max_story_length must be > 0"
            )

        if offset_do >= max_story_length:
            raise ValueError(
                "offset_do must be smaller than max_story_length"
            )

        if min_gap < 1:
            raise ValueError(
                "min_gap must be >= 1"
            )

        if bit_chunk_size < 1:
            raise ValueError(
                "bit_chunk_size must be >= 1"
            )

        self.offset_do = offset_do
        self.max_story_length = max_story_length
        self.min_gap = min_gap
        self.bit_chunk_size = bit_chunk_size

    @staticmethod
    def _normalize_key(
        key: str | bytes | int
    ) -> bytes:

        if isinstance(key, int):
            if key < 0:
                raise ValueError(
                    "integer key must be >= 0"
                )
            key_bytes = str(key).encode("utf-8")
        elif isinstance(key, str):
            key_bytes = key.encode("utf-8")
        elif isinstance(key, bytes):
            key_bytes = key
        else:
            raise TypeError("key must be str, bytes, or int")

        if not key_bytes:
            raise ValueError("key must not be empty")

        return key_bytes

    @staticmethod
    def _shake128_value(
        key_material: bytes,
        counter: int,
    ) -> int:

        if counter < 0:
            raise ValueError(
                "counter must be >= 0"
            )

        counter_bytes = counter.to_bytes(
            8,
            byteorder="big",
            signed=False,
        )

        domain = b"LLM-SHIELD-POSITION-V1"

        data = (
            domain
            + b"|"
            + key_material
            + b"|"
            + counter_bytes
        )

        digest = hashlib.shake_128(
            data
        ).digest(16)

        return int.from_bytes(
            digest,
            byteorder="big",
            signed=False,
        )

    def _step_size(
        self,
        key_material: bytes,
        counter: int,
    ) -> int:
        chunk_range = 1 << self.bit_chunk_size
        return self.offset_do + (
            self._shake128_value(key_material, counter)
            % chunk_range
        )

    def generate(
        self,
        number_of_positions: int = 0,
        key_material: str | bytes | int | None = None,
    ) -> list[int]:
        """Generate deterministic positions from key material."""

        if number_of_positions < 0:

            raise ValueError(
                "number_of_positions must be >= 0"
            )

        if number_of_positions == 0:

            return []

        if key_material is None:
            raise ValueError("key_material is required")

        key_material_bytes = self._normalize_key(key_material)

        available_length = (
            self.max_story_length
            - self.offset_do
        )

        max_possible = (
            (available_length - 1)
            // self.min_gap
        ) + 1

        if number_of_positions > max_possible:

            raise ValueError(
                "Cannot generate the requested number of "
                "positions with the current min_gap. "
                f"Requested={number_of_positions}, "
                f"maximum_possible={max_possible}, "
                f"min_gap={self.min_gap}, "
                f"available_length={available_length}."
            )

        positions: list[int] = []
        position = self.offset_do

        for counter in range(number_of_positions):
            position += self._step_size(key_material_bytes, counter)

            if position >= self.max_story_length:
                raise ValueError(
                    "Generated positions exceed max_story_length. "
                    "Increase max_story_length or reduce the number "
                    "of positions."
                )

            if positions and position - positions[-1] < self.min_gap:
                raise ValueError(
                    "Generated position gap is smaller than min_gap."
                )

            positions.append(position)

        return positions

    def generate_for_message(
        self,
        message_length: int = 0,
        key_material: str | bytes | int | None = None,
    ) -> list[int]:
        """Generate one position per message character."""

        if message_length < 0:

            raise ValueError(
                "message_length must be >= 0"
            )

        return self.generate(
            number_of_positions=message_length,
            key_material=key_material,
        )


def generate_positions(
    key_material: str | bytes | int | None = None,
    number_of_positions: int = 0,
    *,
    offset_do: int = 32,
    max_story_length: int = 100_000,
    min_gap: int = 20,
    bit_chunk_size: int = 5,
) -> list[int]:
    """Generate positions using the functional API."""

    generator = PositionGenerator(
        offset_do=offset_do,
        max_story_length=max_story_length,
        min_gap=min_gap,
        bit_chunk_size=bit_chunk_size,
    )

    return generator.generate(
        key_material=key_material,
        number_of_positions=number_of_positions,
    )


__all__ = [
    "PositionGenerator"
]
