"""Character mapping for the Phase 1 steganography prototype.

The project uses the 16-character h4 alphabet:

    SPACE E T A O N I S R H D L U C M F

The public mapping is reversible for arbitrary UTF-8 text. Each byte is
encoded as two h4 characters, so the output always stays inside the h4
alphabet while still round-tripping to the original input.

This module stays independent from the LLM and position-generation
components.
"""


from __future__ import annotations

from typing import Union


class CharacterMap:
    """Reversible h4-based byte mapping."""
    ALPHABET = " ETAONISRHDLUCMF"

    # Create reverse lookup table.
    CHAR_TO_VALUE = {
        character: value
        for value, character in enumerate(ALPHABET)
    }

    VALUE_TO_CHAR = {
        value: character
        for value, character in enumerate(ALPHABET)
    }

    def encode(self, data: Union[str, bytes]) -> str:
        """Map input text or bytes to the h4 alphabet."""

        if isinstance(data, bytes):
            raw = data
        elif isinstance(data, str):
            raw = data.encode("utf-8")
        else:
            raise TypeError(
                "data must be a str or bytes object"
            )

        if not raw:
            return ""

        encoded: list[str] = []

        for byte in raw:

            encoded.append(
                self.VALUE_TO_CHAR[(byte >> 4) & 0x0F]
            )
            encoded.append(
                self.VALUE_TO_CHAR[byte & 0x0F]
            )

        return "".join(encoded)

    def decode(self, characters: str) -> str:
        """Reverse an h4-mapped character sequence."""

        if not isinstance(characters, str):
            raise TypeError(
                "characters must be a string"
            )

        if not characters:
            return ""

        if len(characters) % 2 != 0:
            raise ValueError(
                "Encoded h4 text must contain an even number of characters."
            )

        data = bytearray()

        for index in range(0, len(characters), 2):

            high_char = characters[index]
            low_char = characters[index + 1]

            if high_char not in self.CHAR_TO_VALUE:
                raise ValueError(
                    f"Invalid h4 character: {high_char!r}"
                )

            if low_char not in self.CHAR_TO_VALUE:
                raise ValueError(
                    f"Invalid h4 character: {low_char!r}"
                )

            high = self.CHAR_TO_VALUE[high_char]
            low = self.CHAR_TO_VALUE[low_char]

            data.append((high << 4) | low)

        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "Encoded h4 text does not decode to valid UTF-8."
            ) from exc

    def to_values(self, characters: str) -> list[int]:
        """Convert h4 characters to their 4-bit integer values."""

        if not isinstance(characters, str):
            raise TypeError(
                "characters must be a string"
            )

        values: list[int] = []

        for character in characters.upper():

            if character not in self.CHAR_TO_VALUE:

                raise ValueError(
                    f"Character {character!r} "
                    "is not part of the h4 alphabet."
                )

            values.append(
                self.CHAR_TO_VALUE[character]
            )

        return values

    def from_values(self, values: list[int]) -> str:
        """Convert 4-bit integer values back to h4 characters."""

        if not isinstance(values, list):
            raise TypeError(
                "values must be a list"
            )

        characters: list[str] = []

        for value in values:

            if not isinstance(value, int):
                raise TypeError(
                    "Each h4 value must be an integer."
                )

            if value < 0 or value > 15:
                raise ValueError(
                    f"h4 value must be between 0 and 15: {value}"
                )

            characters.append(
                self.VALUE_TO_CHAR[value]
            )

        return "".join(characters)


