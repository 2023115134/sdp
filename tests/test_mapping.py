import pytest

from app.crypto.mapping import CharacterMap
from app.llm.generator import LLMGenerator


@pytest.mark.parametrize(
    "payload",
    [
        "HELLO",
        "A",
        "123456",
        "secret-message",
        "",
    ],
)
def test_mapping_round_trip(payload):
    mapper = CharacterMap()
    encoded = mapper.encode(payload)
    decoded = mapper.decode(encoded)
    assert decoded == payload


def test_mapping_rejects_invalid_characters():
    mapper = CharacterMap()
    with pytest.raises(ValueError):
        mapper.decode("G")


def test_value_conversion_round_trip():
    mapper = CharacterMap()
    mapped = mapper.encode("HELLO")

    assert mapper.from_values(mapper.to_values(mapped)) == mapped


def test_generator_model_error_is_actionable():
    generator = LLMGenerator(model_name="definitely-not-a-real-model-name-xyz")
    with pytest.raises(RuntimeError, match="LLM_MODEL_NAME|cached local model|internet access"):
        generator.generate("hello world", max_new_tokens=2)
