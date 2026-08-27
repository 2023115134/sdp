"""LLM-SHIELD paper-based fixed-position end-to-end demo."""

from __future__ import annotations

import logging
import time

from app.crypto.mapping import CharacterMap
from app.crypto.position_generator import PositionGenerator
from app.extraction.extractor import Extractor
from app.llm.embedder import EmbedderLLM
from app.llm.generator import LLMGenerator


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)


def main() -> None:

    print("=" * 70)
    print("LLM-SHIELD END-TO-END TEST")
    print("=" * 70)

    # ==============================================================
    # 1. USER INPUT
    # ==============================================================

    secret = input("Enter secret message: ").strip()

    if not secret:
        raise ValueError("Secret message cannot be empty.")

    while True:
        topic = input("Enter topic: ").strip()

        if topic:
            break

        print("Topic cannot be empty. Please enter a topic.")

    # ==============================================================
    # 2. ORIGINAL SECRET
    # ==============================================================

    print("\nORIGINAL SECRET")
    print("-" * 70)
    print(secret)

    # ==============================================================
    # 3. CHARACTER MAPPING
    # ==============================================================

    character_map = CharacterMap()
    mapped = character_map.encode(secret)

    print("\nMAPPED SECRET")
    print("-" * 70)
    print(mapped)

    # ==============================================================
    # 4. FIXED POSITION GENERATION
    # ==============================================================

    print("\nPOSITION GENERATION")
    print("-" * 70)

    # Same key must be used for embedding and extraction.
    key = "test-secret-key"

    position_generator = PositionGenerator(
        offset_do=32,
        max_story_length=1000,
    )

    positions = position_generator.generate_for_message(
        key=key,
        message_length=len(mapped),
    )

    print("Secret length:", len(secret))
    print("Mapped length:", len(mapped))
    print("Positions:", positions)

    # ==============================================================
    # 5. LLM INITIALIZATION
    # ==============================================================

    print("\nLLM INITIALIZATION")
    print("-" * 70)

    generator = LLMGenerator()

    print("Model:", generator.model_name)
    print("Device:", generator.device)

    # ==============================================================
    # 6. LLM STEGANOGRAPHIC EMBEDDING
    # ==============================================================

    print("\nLLM STEGANOGRAPHIC EMBEDDING")
    print("-" * 70)

    print("Topic:", topic)
    print("Embedding mode: fixed (paper)")

    embedder = EmbedderLLM(
        llm_generator=generator,
        character_map=character_map,
    )

    embedding_started = time.perf_counter()

    result = embedder.embed(
        topic=topic,
        characters=mapped,
        positions=positions,
        temperature=0.70,
        top_k=40,
        max_new_tokens=8,
        max_attempts=4000,
        max_retries=6,
    )

    embedding_seconds = time.perf_counter() - embedding_started

    cover_text = result.story

    # Keep the positions returned by the embedder.
    positions = result.positions

    # ==============================================================
    # 7. GENERATED COVER TEXT
    # ==============================================================

    print("\nGENERATED COVER TEXT")
    print("=" * 70)
    print(cover_text)

    print("\nCOVER LENGTH")
    print("-" * 70)
    print(len(cover_text))

    print("\nEMBEDDING POSITIONS")
    print("-" * 70)
    print(positions)

    print("\nEMBEDDED CHARACTERS")
    print("-" * 70)
    print(result.embedded_characters)

    print("\nCANDIDATE EVALUATIONS")
    print("-" * 70)
    print(result.attempts)

    # ==============================================================
    # 8. NATURALNESS VALIDATION
    # ==============================================================

    naturalness = embedder._validate_cover_naturalness(
        cover_text,
        topic,
    )

    print("\nNATURALNESS VALIDATION")
    print("-" * 70)

    print(
        "Topic relevance:",
        "PASS" if naturalness["topic_relevance"] else "WARNING",
    )

    print(
        "Repetition check:",
        "PASS" if naturalness["repetition"] else "WARNING",
    )

    print(
        "Sentence completeness:",
        "PASS" if naturalness["sentence_completeness"] else "WARNING",
    )

    print(
        "Malformed/technical:",
        "PASS" if naturalness["malformed_or_technical"] else "WARNING",
    )

    # ==============================================================
    # 9. EXTRACTION
    # ==============================================================

    print("\nEXTRACTION")
    print("-" * 70)

    extraction_started = time.perf_counter()

    extractor = Extractor(
        position_generator=position_generator,
        character_map=character_map,
    )

    extracted = extractor.recover(
        cover_text=cover_text,
        key=key,
        message_length=len(mapped),
    )

    extraction_seconds = time.perf_counter() - extraction_started

    print("Extracted mapped message:", extracted.characters)
    print("Recovered secret:", extracted.message)

    # ==============================================================
    # 10. END-TO-END VERIFICATION
    # ==============================================================

    embedding_match = embedder.verify_embedding(
        cover_text=cover_text,
        mapped=mapped,
        positions=positions,
    )
    mapped_match = extracted.characters.upper() == mapped.upper()
    secret_match = extracted.message == secret

    # IMPORTANT:
    # Naturalness should be reported separately.
    # It should NOT break cryptographic extraction verification.
    end_to_end_pass = mapped_match and secret_match

    print("\nEND-TO-END VERIFICATION")
    print("-" * 70)

    print("Original secret :", secret)
    print("Recovered secret:", extracted.message)
    print("Fixed positions :", positions)
    print("Embedding check  :", "PASS" if embedding_match else "FAIL")

    print(
        "Secret match    :",
        "PASS" if secret_match else "FAIL",
    )

    print("Mapped secret   :", mapped)
    print("Extracted mapped:", extracted.characters)

    print(
        "Mapped match    :",
        "PASS" if mapped_match else "FAIL",
    )

    # ==============================================================
    # 11. PERFORMANCE
    # ==============================================================

    print("\nPERFORMANCE")
    print("-" * 70)

    print(
        f"Embedding time: {embedding_seconds:.2f} seconds"
    )

    print(
        f"Extraction time: {extraction_seconds:.4f} seconds"
    )

    print(
        "Candidate evaluations:",
        result.attempts,
    )

    # ==============================================================
    # 12. FINAL RESULT
    # ==============================================================

    print("\nFINAL RESULT")
    print("=" * 70)

    if end_to_end_pass:
        print("END-TO-END TEST: PASS")
    else:
        print("END-TO-END TEST: FAIL")

    print("=" * 70)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()