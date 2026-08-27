"""Extraction component for the Phase 1 LLM steganography prototype.

The extractor supports two related workflows:

1. Direct extraction from a cover text and explicit positions.
2. Full recovery from a cover text, key, and message length using the
   same deterministic position generator as the sender.

The extractor never modifies the cover text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from app.crypto.mapping import CharacterMap
from app.crypto.position_generator import PositionGenerator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractionResult:
    """Result returned by the extraction process."""

    characters: str
    message: str
    positions: list[int]


class Extractor:
    """Recover hidden characters from an LLM-generated cover text."""

    def __init__(
        self,
        position_generator: PositionGenerator | None = None,
        character_map: CharacterMap | None = None,
    ) -> None:

        self.position_generator = (
            position_generator
            if position_generator is not None
            else PositionGenerator()
        )

        self.character_map = (
            character_map
            if character_map is not None
            else CharacterMap()
        )

    # ================================================================
    # READ CHARACTERS
    # ================================================================

    @staticmethod
    def _read_characters(
        cover_text: str,
        positions: Sequence[int],
        strict: bool = True,
    ) -> str:
        """Read characters from the cover text at target positions."""

        if not isinstance(cover_text, str):
            raise TypeError(
                "cover_text must be a string"
            )

        characters: list[str] = []

        for position in positions:

            if position < 0:
                if strict:
                    raise ValueError(
                        f"Invalid position: {position}"
                    )
                continue

            if position >= len(cover_text):
                if strict:
                    raise ValueError(
                        "Position is outside cover text: "
                        f"position={position}, "
                        f"cover_length={len(cover_text)}"
                    )
                continue

            characters.append(
                cover_text[position]
            )

        return "".join(characters)

    # ================================================================
    # DIRECT EXTRACTION
    # ================================================================

    @staticmethod
    def extract(
        cover_text: str,
        positions: Sequence[int],
    ) -> str:
        """Extract characters directly from explicit positions."""

        return Extractor._read_characters(
            cover_text=cover_text,
            positions=positions,
            strict=False,
        )

    # ================================================================
    # FULL RECOVERY
    # ================================================================

    def recover(
        self,
        cover_text: str,
        key: str | bytes | int,
        message_length: int,
        positions: Sequence[int] | None = None,
    ) -> ExtractionResult:
        """
        Recover the hidden message using the configured generator and map.

        Args:
            cover_text:
                LLM-generated cover text.

            key:
                Same key used by PositionGenerator.

            message_length:
                Number of embedded h4 characters.

        Returns:
            ExtractionResult.
        """

        if not cover_text:
            raise ValueError(
                "cover_text must not be empty"
            )

        if message_length < 0:
            raise ValueError(
                "message_length must be >= 0"
            )

        # Dynamic mode supplies the positions recorded by the sender. The
        # legacy path continues to derive positions from the shared key.
        if positions is None:
            positions = self.position_generator.generate_for_message(
                key=key,
                message_length=message_length,
            )
        elif len(positions) != message_length:
            raise ValueError(
                "Number of positions must equal message_length."
            )

        logger.info(
            "Generated %d extraction positions",
            len(positions),
        )

        # ------------------------------------------------------------
        # Read characters from cover text.
        # ------------------------------------------------------------

        characters = self._read_characters(
            cover_text=cover_text,
            positions=positions,
            strict=True,
        ).upper()

        logger.info(
            "Extracted characters: %r",
            characters,
        )

        # ------------------------------------------------------------
        # Convert h4 characters back to message.
        # ------------------------------------------------------------

        message = self.character_map.decode(characters)

        logger.info(
            "Recovered message: %r",
            message,
        )

        return ExtractionResult(
            characters=characters,
            message=message,
            positions=positions,
        )

    # ================================================================
    # BACKWARD COMPATIBILITY
    # ================================================================

    def extract_with_key(
        self,
        cover_text: str,
        key: str | bytes | int,
        message_length: int,
        positions: Sequence[int] | None = None,
    ) -> ExtractionResult:
        """Compatibility alias for callers that prefer an explicit name."""

        return self.recover(
            cover_text=cover_text,
            key=key,
            message_length=message_length,
            positions=positions,
        )


# ====================================================================
# SIMPLE TEST
# ====================================================================

def _run_test() -> None:
    """Run a basic extractor test."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    print("=" * 70)
    print("EXTRACTOR TEST")
    print("=" * 70)

    generator = PositionGenerator(
        offset_do=32,
        max_story_length=1000,
    )

    extractor = Extractor(
        position_generator=generator,
    )

    key = "test-secret-key"
    secret = "HELLO"
    hidden = CharacterMap().encode(secret)

    # ------------------------------------------------------------
    # Create a synthetic cover text.
    #
    # This test checks extraction independently from the LLM.
    # ------------------------------------------------------------

    positions = generator.generate_for_message(
        key=key,
        message_length=len(hidden),
    )

    print("\nPositions:")
    print(positions)

    cover = list(
        "A" * 200
    )

    required_length = max(positions) + 1

    if len(cover) < required_length:
        cover.extend(
            ["A"] * (
                required_length - len(cover)
            )
        )

    for position, character in zip(positions, hidden):

        cover[position] = character

    cover_text = "".join(cover)

    print("\nHidden message:")
    print(hidden)

    print("\nOriginal message:")
    print(secret)

    print("\nCover length:")
    print(len(cover_text))

    # ------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------

    try:

        result = extractor.recover(
            cover_text=cover_text,
            key=key,
            message_length=len(hidden),
        )

        print("\nExtracted characters:")
        print(result.characters)

        print("\nRecovered message:")
        print(result.message)

        print("\nPositions:")
        print(result.positions)

        print("\nOriginal message:")
        print(secret)

        print(
            "\nRound-trip:",
            result.message.upper()
            == secret.upper(),
        )

    except Exception as exc:

        print("\nEXTRACTION TEST FAILED")
        print(
            f"Error: {exc}"
        )


if __name__ == "__main__":
    _run_test()


__all__ = [
    "Extractor",
    "ExtractionResult",
]
