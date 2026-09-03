"""Small SHAKE128 position and Qwen embedding demo."""

from __future__ import annotations

import time

from app.crypto.mapping import CharacterMap
from app.crypto.position_generator import generate_positions
from app.extraction.extractor import Extractor
from app.llm.embedder import EmbedderLLM
from app.llm.generator import LLMGenerator


def main() -> None:
    print("=" * 72)
    print("SHAKE128 EMBEDDER DEMO")
    print("=" * 72)

    topic = input("Topic : ").strip() 
    secret = input("Secret character sequence : ").strip() 

    character_map = CharacterMap()
    mapped_secret = character_map.encode(secret)

    offset_do = 32
    bit_chunk_size = 5
    maximum_step = offset_do + ((1 << bit_chunk_size) - 1)
    max_story_length = max(
        512,
        offset_do + len(mapped_secret) * maximum_step + 1,
    )
    positions = generate_positions(
        key_material=secret,
        number_of_positions=len(mapped_secret),
        offset_do=offset_do,
        max_story_length=max_story_length,
        min_gap=1,
        bit_chunk_size=bit_chunk_size,
    )

    print("\nTopic:", topic)
    print("Input secret:", repr(secret))
    print("Mapped secret sequence:", repr(mapped_secret))
    print("SHAKE128 positions:", positions)
    print("Character positions:")
    for character, position in zip(mapped_secret, positions):
        display_character = "SPACE" if character == " " else character
        print(f"  {display_character!r} -> {position}")

    print("\nEmbedding with EmbedderLLM and Qwen...")
    embedder = EmbedderLLM(
        llm_generator=LLMGenerator(),
        character_map=character_map,
    )
    started = time.perf_counter()
    result = embedder.embed(
        topic=topic,
        characters=mapped_secret,
        positions=positions,
        max_new_tokens=32,
        max_attempts=2500,
        max_retries=3,
    )
    embedding_seconds = time.perf_counter() - started

    print("\nGenerated cover story:")
    print(result.story)
    print(f"\nTotal embedding time: {embedding_seconds:.2f} seconds")

    extractor = Extractor(character_map=character_map)
    extracted_sequence = extractor.extract(
        cover_text=result.story,
        positions=positions,
    ).upper()
    extracted_secret = character_map.decode(extracted_sequence)
    matches = (
        extracted_sequence == mapped_secret
        and extracted_secret == secret
    )

    print("\nExtracted characters:", repr(extracted_sequence))
    print("Decoded extracted sequence:", repr(extracted_secret))
    print("Input matches extracted:", matches)
    print("=" * 72)
    print("RESULT:", "PASS" if matches else "FAIL")
    print("=" * 72)


if __name__ == "__main__":
    main()
