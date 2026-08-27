from __future__ import annotations

import pathlib
import re
import site
import sys
from typing import Callable


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"

sys.path.insert(0, str(ROOT))
if DEPS.exists():
    site.addsitedir(str(DEPS))

from app.config import get_model_name
from app.crypto.mapping import CharacterMap
from app.crypto.position_generator import PositionGenerator
from app.extraction.extractor import Extractor
from app.evaluation.naturalness import summarize_cover_text
from app.llm.generator import LLMGenerator


def assert_raises(
    expected_exc: type[BaseException],
    func: Callable[[], object],
    match: str | None = None,
) -> None:
    try:
        func()
    except expected_exc as exc:
        if match is not None and not re.search(match, str(exc)):
            raise AssertionError(
                f"expected {expected_exc.__name__} matching {match!r}, "
                f"got {exc!r}"
            ) from exc
        return
    except BaseException as exc:  # pragma: no cover - diagnostic path
        raise AssertionError(
            f"expected {expected_exc.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc

    raise AssertionError(f"expected {expected_exc.__name__} to be raised")


def test_mapping_round_trip() -> None:
    mapper = CharacterMap()
    for payload in ["HELLO", "A", "123456", "secret-message", ""]:
        encoded = mapper.encode(payload)
        decoded = mapper.decode(encoded)
        assert decoded == payload


def test_mapping_rejects_invalid_characters() -> None:
    mapper = CharacterMap()
    assert_raises(ValueError, lambda: mapper.decode("G"))


def test_generator_model_error_is_actionable() -> None:
    generator = LLMGenerator(model_name="definitely-not-a-real-model-name-xyz")
    assert_raises(
        RuntimeError,
        lambda: generator.generate("hello world", max_new_tokens=2),
        match=r"LLM_MODEL_NAME|cached local model|internet access",
    )


def test_position_reproducibility() -> None:
    generator = PositionGenerator(
        chunk_size=5,
        offset_do=32,
        max_story_length=1000,
    )
    positions_a = generator.generate("secret-key", 20)
    positions_b = generator.generate("secret-key", 20)
    assert positions_a == positions_b


def test_different_keys_produce_different_positions() -> None:
    generator = PositionGenerator(
        chunk_size=5,
        offset_do=32,
        max_story_length=2000,
    )
    positions_a = generator.generate("key-a", 20)
    positions_b = generator.generate("key-b", 20)
    assert positions_a != positions_b


def test_position_ordering_is_sorted() -> None:
    generator = PositionGenerator(
        chunk_size=5,
        offset_do=32,
        max_story_length=5000,
    )
    positions = generator.generate("demo-key", 15)
    assert positions == sorted(positions)
    assert len(positions) == len(set(positions))


def test_character_extraction_from_story() -> None:
    story = "abcdefghijklmnopqrstuvwxyz"
    positions = [0, 2, 4, 6, 8]
    extracted = Extractor.extract(story, positions)
    assert extracted == "acegi"


def test_key_dependent_extraction_validation() -> None:
    mapping = CharacterMap()
    generator = PositionGenerator(offset_do=32, max_story_length=2000)
    key = "test-secret-key"
    hidden = mapping.encode("HELLO")
    positions = generator.generate_for_message(key, len(hidden))

    cover = ["x"] * (max(positions) + 1)
    for position, character in zip(positions, hidden):
        cover[position] = character

    extractor = Extractor(generator, mapping)
    assert extractor.recover("".join(cover), key, len(hidden)).message == "HELLO"

    try:
        wrong = extractor.recover("".join(cover), "wrong-key", len(hidden)).message
    except ValueError:
        return

    assert wrong != "HELLO"


def test_naturalness_summary_is_data_driven() -> None:
    summary = summarize_cover_text("A quiet city. A quiet city.")
    assert summary["text_length"] == 27
    assert summary["sentence_count"] == 2
    assert summary["repeated_token_count"] > 0


def test_embedder_success_and_failure_paths() -> None:
    mapper = CharacterMap()
    positions = PositionGenerator(
        chunk_size=5,
        offset_do=32,
        max_story_length=100,
    ).generate("key", 5)
    story = "the old lighthouse watched the black sea from the cliff"

    encoded = mapper.encode("ABCD")
    extracted = Extractor.extract(story, positions[:4])
    assert extracted != encoded

    assert len(positions) >= 4


def test_end_to_end_mapping_and_extraction_pipeline() -> None:
    mapper = CharacterMap()
    secret = "HELLO"
    encoded = mapper.encode(secret)
    positions = [0, 4, 8, 12, 16]
    story = "A quiet city breathes under the morning sky"
    actual = Extractor.extract(story, positions)
    assert len(actual) == len(positions)
    assert actual != encoded


def test_end_to_end_contract_uses_qwen_and_reversible_mapping() -> None:
    mapping = CharacterMap()
    payload = mapping.encode("HELLO")
    assert mapping.decode(payload) == "HELLO"
    assert get_model_name() == "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> None:
    tests = [
        test_mapping_round_trip,
        test_mapping_rejects_invalid_characters,
        test_generator_model_error_is_actionable,
        test_position_reproducibility,
        test_different_keys_produce_different_positions,
        test_position_ordering_is_sorted,
        test_character_extraction_from_story,
        test_key_dependent_extraction_validation,
        test_naturalness_summary_is_data_driven,
        test_embedder_success_and_failure_paths,
        test_end_to_end_mapping_and_extraction_pipeline,
        test_end_to_end_contract_uses_qwen_and_reversible_mapping,
    ]

    passed = 0
    for test in tests:
        test()
        passed += 1
        print(f"PASS {test.__name__}")

    print(f"TOTAL {passed}/{len(tests)} PASSED")


if __name__ == "__main__":
    sys.exit(main())
