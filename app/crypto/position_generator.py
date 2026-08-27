"""
Deterministic SHAKE128-based position generation.

Phase 1 implementation for the LLM steganography prototype.

Positions are generated deterministically from the key using SHAKE128.

For LLM-based embedding, a minimum gap can be enforced between
embedding positions so that consecutive secret characters are not
too close together for the language model to satisfy.
"""

from __future__ import annotations

import hashlib


class PositionGenerator:
    """Generate deterministic, unique embedding positions."""

    def __init__(
        self,
        chunk_size: int | None = None,
        offset_do: int = 32,
        max_story_length: int = 100_000,
        min_gap: int = 20,
        bit_chunk_size: int = 5,
    ) -> None:

        if chunk_size is not None:

            if chunk_size < 1:
                raise ValueError(
                    "chunk_size must be >= 1"
                )

            # Backwards-compatible alias used by older callers and tests.
            min_gap = chunk_size

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

        self.chunk_size = (
            chunk_size
            if chunk_size is not None
            else min_gap
        )
        self.offset_do = offset_do
        self.max_story_length = max_story_length
        self.min_gap = min_gap
        self.bit_chunk_size = bit_chunk_size
        self._legacy_sampling = chunk_size is not None

    # ==================================================================
    # KEY NORMALIZATION
    # ==================================================================

    @staticmethod
    def _normalize_key(
        key: str | bytes | int
    ) -> bytes:

        if isinstance(key, bytes):

            if not key:
                raise ValueError(
                    "key must not be empty"
                )

            return key

        if isinstance(key, str):

            if not key:
                raise ValueError(
                    "key must not be empty"
                )

            return key.encode("utf-8")

        if isinstance(key, int):

            if key < 0:
                raise ValueError(
                    "integer key must be >= 0"
                )

            return str(key).encode("utf-8")

        raise TypeError(
            "key must be str, bytes, or int"
        )

    # ==================================================================
    # SHAKE128
    # ==================================================================

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

    # ==================================================================
    # SINGLE POSITION
    # ==================================================================

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

    def _candidate_position(
        self,
        key_material: bytes,
        counter: int,
    ) -> int:
        available_range = self.max_story_length - self.offset_do
        return self.offset_do + (
            self._shake128_value(key_material, counter)
            % available_range
        )

    # ==================================================================
    # POSITION VALIDATION
    # ==================================================================

    def _is_valid_gap(
        self,
        candidate: int,
        positions: list[int],
    ) -> bool:
        """
        Check whether candidate maintains the required
        minimum distance from all existing positions.
        """

        for position in positions:

            if abs(candidate - position) < self.min_gap:

                return False

        return True

    # ==================================================================
    # POSITION GENERATION
    # ==================================================================

    def generate(
        self,
        key: str | bytes | int,
        number_of_positions: int,
    ) -> list[int]:
        """
        Generate deterministic unique embedding positions.

        The positions are generated using SHAKE128 and must satisfy
        the configured minimum gap.
        """

        if number_of_positions < 0:

            raise ValueError(
                "number_of_positions must be >= 0"
            )

        if number_of_positions == 0:

            return []

        key_material = self._normalize_key(
            key
        )

        if self._legacy_sampling:
            positions: list[int] = []
            counter = 0
            while len(positions) < number_of_positions:
                candidate = self._candidate_position(
                    key_material,
                    counter,
                )
                counter += 1
                if self._is_valid_gap(candidate, positions):
                    positions.append(candidate)
            return sorted(positions)

        available_length = (
            self.max_story_length
            - self.offset_do
        )

        # --------------------------------------------------------------
        # Maximum number of positions possible with min_gap.
        # --------------------------------------------------------------

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
            position += self._step_size(key_material, counter)

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

    # ==================================================================
    # MESSAGE HELPER
    # ==================================================================

    def generate_for_message(
        self,
        key: str | bytes | int,
        message_length: int,
    ) -> list[int]:
        """
        Generate exactly one position per embedded character.
        """

        if message_length < 0:

            raise ValueError(
                "message_length must be >= 0"
            )

        return self.generate(
            key=key,
            number_of_positions=message_length,
        )


# ======================================================================
# TESTS
# ======================================================================

def _run_tests() -> None:

    print("=" * 70)
    print("SHAKE128 POSITION GENERATOR TEST")
    print("=" * 70)

    generator = PositionGenerator(
        offset_do=32,
        max_story_length=1000,
        min_gap=20,
    )

    key = "test-secret-key"

    # ------------------------------------------------------------------
    # Generate positions
    # ------------------------------------------------------------------

    positions = generator.generate(
        key=key,
        number_of_positions=5,
    )

    print("\nGenerated positions:")
    print(positions)

    # ------------------------------------------------------------------
    # Check number
    # ------------------------------------------------------------------

    print(
        "\nNumber of positions:",
        len(positions),
    )

    # ------------------------------------------------------------------
    # Check uniqueness
    # ------------------------------------------------------------------

    unique = (
        len(positions)
        == len(set(positions))
    )

    print(
        "All positions unique:",
        unique,
    )

    # ------------------------------------------------------------------
    # Check range
    # ------------------------------------------------------------------

    valid_range = all(
        generator.offset_do
        <= position
        < generator.max_story_length
        for position in positions
    )

    print(
        "All positions in valid range:",
        valid_range,
    )

    # ------------------------------------------------------------------
    # Check minimum gap
    # ------------------------------------------------------------------

    gaps = [
        positions[i + 1] - positions[i]
        for i in range(len(positions) - 1)
    ]

    gap_test = all(
        gap >= generator.min_gap
        for gap in gaps
    )

    print(
        "Gaps:",
        gaps,
    )

    print(
        "Minimum gap satisfied:",
        gap_test,
    )

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    positions_again = generator.generate(
        key=key,
        number_of_positions=5,
    )

    deterministic = (
        positions
        == positions_again
    )

    print(
        "Same key gives same positions:",
        deterministic,
    )

    # ------------------------------------------------------------------
    # Different key
    # ------------------------------------------------------------------

    different_positions = generator.generate(
        key="different-secret-key",
        number_of_positions=5,
    )

    different_key = (
        positions
        != different_positions
    )

    print(
        "Different key changes positions:",
        different_key,
    )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    all_tests_passed = all(
        [
            unique,
            valid_range,
            gap_test,
            deterministic,
            different_key,
        ]
    )

    print("\n" + "=" * 70)

    if all_tests_passed:

        print(
            "POSITION GENERATOR TEST: PASS"
        )

    else:

        print(
            "POSITION GENERATOR TEST: FAIL"
        )

    print("=" * 70)


if __name__ == "__main__":
    _run_tests()


__all__ = [
    "PositionGenerator"
]
