from types import SimpleNamespace

import pytest

from app.crypto.mapping import CharacterMap
from app.web_demo import DemoPipelineError, run_receiver, run_sender


@pytest.fixture
def fake_embedder(monkeypatch):
    from app.llm import embedder as embedder_module
    from app.llm import generator as generator_module

    class FakeGenerator:
        model_name = "test-model (explicitly mocked in test)"

    class FakeEmbedder:
        def __init__(self, llm_generator):
            self._embedding_stats = {
                "llm_calls": 0,
                "candidate_evaluations": 0,
                "retries": 0,
            }

        def embed(self, *, characters, positions, **kwargs):
            story = ["x"] * (max(positions) + 1)
            for character, position in zip(characters, positions):
                story[position] = character
            return SimpleNamespace(
                story="".join(story),
                positions=list(positions),
                attempts=0,
            )

        @staticmethod
        def verify_embedding(story, characters, positions):
            return all(
                story[position].upper() == character.upper()
                for character, position in zip(characters, positions)
            )

    monkeypatch.setattr(generator_module, "LLMGenerator", FakeGenerator)
    monkeypatch.setattr(embedder_module, "EmbedderLLM", FakeEmbedder)


def test_sender_and_receiver_use_real_crypto_and_safe_output_shape(fake_embedder):
    sender = run_sender(
        message="hello",
        password="correct horse",
        topic="A quiet garden",
    )

    assert sender["mode"] == "PSK"
    assert "correct horse" not in str(sender)
    assert "dk1" not in sender["stages"]["pbkdf2"]
    assert sender["stages"]["encryption"]["tag_length"] == 16
    assert sender["stages"]["payload"]["order"] == "tag || ciphertext"
    assert sender["stages"]["mapping"]["output_character_length"] == sender["session"]["mapped_length"]
    assert sender["stages"]["positions"]["positions"] == sender["stages"]["embedding"]["target_positions"]

    receiver = run_receiver(
        carrier_text=sender["stages"]["carrier"]["text"],
        password="correct horse",
        salt_hex=sender["session"]["salt_hex"],
        nonce_hex=sender["session"]["nonce_hex"],
        mapped_length=sender["session"]["mapped_length"],
    )
    assert receiver["stages"]["authentication"]["status"] == "success"
    assert receiver["stages"]["authentication"]["plaintext"] == "hello"


def test_receiver_authentication_failure_never_returns_plaintext(fake_embedder):
    sender = run_sender(
        message="tamper me",
        password="correct horse",
        topic="A quiet garden",
    )
    carrier = list(sender["stages"]["carrier"]["text"])
    first_position = sender["stages"]["positions"]["positions"][0]
    original = carrier[first_position]
    carrier[first_position] = next(
        character
        for character in CharacterMap.ALPHABET
        if character != original
    )

    receiver = run_receiver(
        carrier_text="".join(carrier),
        password="correct horse",
        salt_hex=sender["session"]["salt_hex"],
        nonce_hex=sender["session"]["nonce_hex"],
        mapped_length=sender["session"]["mapped_length"],
    )
    assert receiver["stages"]["authentication"]["status"] == "failed"
    assert "plaintext" not in receiver["stages"]["authentication"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("message", ""),
        ("password", ""),
        ("topic", ""),
        ("temperature", 0),
        ("top_k", 0),
    ],
)
def test_sender_rejects_invalid_inputs(field, value):
    inputs = {
        "message": "hello",
        "password": "correct horse",
        "topic": "A quiet garden",
    }
    inputs[field] = value
    with pytest.raises(DemoPipelineError):
        run_sender(**inputs)


def test_receiver_rejects_invalid_carrier_and_wrong_password(fake_embedder):
    sender = run_sender(
        message="hello",
        password="correct horse",
        topic="A quiet garden",
    )
    common = {
        "carrier_text": "not a valid carrier",
        "salt_hex": sender["session"]["salt_hex"],
        "nonce_hex": sender["session"]["nonce_hex"],
        "mapped_length": sender["session"]["mapped_length"],
    }
    with pytest.raises(DemoPipelineError):
        run_receiver(password="wrong password", **common)


def test_sender_reports_unavailable_or_failed_llm_without_mock_carrier(monkeypatch):
    from app.llm import embedder as embedder_module
    from app.llm import generator as generator_module

    class FakeGenerator:
        model_name = "test-model"

    class FailedEmbedder:
        def __init__(self, llm_generator):
            pass

        def embed(self, **kwargs):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(generator_module, "LLMGenerator", FakeGenerator)
    monkeypatch.setattr(embedder_module, "EmbedderLLM", FailedEmbedder)

    with pytest.raises(DemoPipelineError, match="Real LLM embedding failed"):
        run_sender(
            message="hello",
            password="correct horse",
            topic="A quiet garden",
        )


def test_receiver_rejects_out_of_range_target_count():
    with pytest.raises(DemoPipelineError, match="Mapped payload length is invalid"):
        run_receiver(
            carrier_text="carrier",
            password="correct horse",
            salt_hex="01020304",
            nonce_hex="00" * 12,
            mapped_length=8194,
        )
