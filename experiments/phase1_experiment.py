"""End-to-end Stage 1 experiment pipeline.

This script demonstrates the prototype flow:
secret -> character mapping -> positions -> LLM embedding -> extraction -> match check.
"""

from __future__ import annotations

import logging
import time

from app.crypto.mapping import CharacterMap
from app.crypto.position_generator import PositionGenerator
from app.extraction.extractor import Extractor
from app.llm.embedder import EmbedderLLM
from app.llm.generator import LLMGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")


def run_experiment() -> None:
    """Run the Phase 1 end-to-end prototype and print diagnostics."""

    secret = "HELLO"
    mapper = CharacterMap()
    encoded = mapper.encode(secret)
    print("Input characters:", secret)
    print("Mapped characters:", encoded)

    position_generator = PositionGenerator(min_gap=5, offset_do=32, max_story_length=2000)
    positions = position_generator.generate(
        number_of_positions=len(encoded),
        key_material="phase1-secret-key",
    )
    print("Generated positions:", positions)

    llm_generator = LLMGenerator()
    embedder = EmbedderLLM(llm_generator=llm_generator, character_map=mapper)

    start = time.time()
    result = embedder.embed(
        topic="Write a short atmospheric story about a city at dawn.",
        characters=encoded,
        positions=positions,
        initial_story="",
        temperature=0.8,
        top_k=40,
        max_new_tokens=60,
        max_attempts=30,
        deterministic=True,
    )
    elapsed = time.time() - start

    story = result.story
    extracted = Extractor.extract(story, positions)
    match = extracted == encoded

    print("Generated story:", story)
    print("Extracted characters:", extracted)
    print("Match:", match)
    print("Embedding attempts:", result.attempts)
    print("Generation parameters:", {"temperature": 0.8, "top_k": 40, "max_new_tokens": 60})
    print("Execution time:", round(elapsed, 4), "seconds")


if __name__ == "__main__":
    run_experiment()
