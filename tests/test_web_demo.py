from types import SimpleNamespace

import pytest

from app.crypto.mapping import CharacterMap
from app.web_demo import (
    DemoPipelineError,
    _install_transformers_weight_progress_hook,
    _log_embedding_progress,
    _restore_transformers_weight_progress_hook,
    run_receiver,
    run_sender,
)


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
            progress_callback = kwargs.get("progress_callback")
            story = ["x"] * (max(positions) + 1)
            if progress_callback is not None:
                progress_callback(0, len(characters), "embedding")
            for index, (character, position) in enumerate(zip(characters, positions)):
                story[position] = character
                if progress_callback is not None:
                    progress_callback(index + 1, len(characters), "embedded")
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
    assert "recovered_plaintext" not in sender["stages"]["sender_verification"]
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


def test_sender_emits_each_actual_stage_output_as_it_becomes_available(fake_embedder):
    emitted = []
    sender = run_sender(
        message="hello",
        password="correct horse",
        topic="A quiet garden",
        on_stage=lambda name, data: emitted.append((name, data)),
    )

    emitted_names = [name for name, _ in emitted]
    assert emitted_names[:7] == [
        "key_establishment",
        "pbkdf2",
        "encryption",
        "payload",
        "mapping",
        "positions",
        "embedding",
    ]
    assert emitted[6][1]["status"] == "running"
    progress_updates = [
        data for name, data in emitted
        if name == "embedding" and "completed_characters" in data
    ]
    assert progress_updates[0]["message"].startswith(
        "Generating candidate for character 1/"
    )
    assert progress_updates[0]["completed_characters"] == 0
    assert progress_updates[-1]["message"].startswith("Embedded character ")
    assert progress_updates[-1]["completed_characters"] == sender["session"]["mapped_length"]
    assert all(update["status"] == "running" for update in progress_updates[:-1])
    assert progress_updates[-1]["status"] == "success"
    assert emitted_names[-3:] == ["embedding", "carrier", "sender_verification"]
    assert emitted[-3][1]["status"] == "success"
    assert emitted[-2][1]["text"] == sender["stages"]["carrier"]["text"]
    assert emitted[-1][1]["extracted"] == sender["stages"]["sender_verification"]["extracted"]


def test_web_sender_logs_real_embedder_events_and_keeps_browser_progress(
    monkeypatch, capsys
):
    from app.llm import generator as generator_module
    from app.llm.embedder import EmbedderLLM

    class FakeGenerator:
        model_name = "test-model"

    def embed_at_target(self, **kwargs):
        self._embedding_stats["candidate_evaluations"] += 2
        return kwargs["story"].ljust(kwargs["position"], " ") + kwargs["character"]

    monkeypatch.setattr(generator_module, "LLMGenerator", FakeGenerator)
    monkeypatch.setattr(EmbedderLLM, "_generate_to_position", embed_at_target)
    monkeypatch.setattr(
        EmbedderLLM,
        "_complete_cover_text",
        lambda self, story, topic, max_tokens=80: story,
    )
    browser_events = []

    sender = run_sender(
        message="x",
        password="test-password",
        topic="A quiet garden",
        on_stage=lambda name, data: browser_events.append((name, data)),
    )

    mapped_count = sender["session"]["mapped_length"]
    progress_updates = [
        data for name, data in browser_events
        if name == "embedding" and "completed_characters" in data
    ]
    assert sender["stages"]["sender_verification"]["authentication"] == "success"
    assert len(progress_updates) > mapped_count
    assert progress_updates[-1]["status"] == "success"
    assert progress_updates[-1]["completed_characters"] == mapped_count
    terminal_output = capsys.readouterr().out
    assert f"Embedding 1/{mapped_count}" in terminal_output
    assert f"Embedding {mapped_count}/{mapped_count}" in terminal_output
    assert "Character: " in terminal_output
    assert "Retry 1: T=0.70 k=20" in terminal_output
    assert "Retry 1/12" not in terminal_output
    assert "candidate evaluations this retry: 2" in terminal_output
    assert "Status: Embedded successfully" in terminal_output
    assert "Character: '" not in terminal_output
    assert "test-password" not in terminal_output


def test_transformers_weight_progress_hook_reports_actual_updates(
    monkeypatch, capsys
):
    from transformers.utils import logging as transformers_logging

    if not callable(getattr(transformers_logging, "set_tqdm_hook", None)):
        pytest.skip("Installed Transformers does not expose a tqdm hook.")

    installed_hooks = []

    def set_tqdm_hook(hook):
        previous = installed_hooks[-1] if installed_hooks else None
        installed_hooks[:] = [] if hook is None else [hook]
        return previous

    monkeypatch.setattr(transformers_logging, "set_tqdm_hook", set_tqdm_hook)
    events = []
    hook_state = _install_transformers_weight_progress_hook(events.append)
    assert hook_state is not None

    class Progress:
        def __init__(self, iterable, total, desc):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.n = 0
            self.disable = False

        def update(self, amount=1):
            self.n += amount

        def close(self):
            pass

    try:
        wrapped = installed_hooks[0](
            Progress,
            ([object(), object()],),
            {"total": 2, "desc": "Loading weights"},
        )
        assert list(wrapped) == [wrapped.iterable[0], wrapped.iterable[1]]
        assert [
            (event["current"], event["total"])
            for event in events
        ] == [(1, 2), (2, 2)]
        for event in events:
            _log_embedding_progress(event)
        _log_embedding_progress(
            {
                "event": "character_embedded",
                "candidate_evaluations": 1,
            }
        )
    finally:
        _restore_transformers_weight_progress_hook(hook_state)

    output = capsys.readouterr().out
    assert "Loading weights: 1/2" in output
    assert "Loading weights: 2/2" in output
    assert "Status: Embedded successfully" in output


def test_sender_emits_retry_progress_without_reporting_completion(monkeypatch):
    from app.llm import embedder as embedder_module
    from app.llm import generator as generator_module

    class FakeGenerator:
        model_name = "test-model"

    class FailedEmbedder:
        def __init__(self, llm_generator):
            self._embedding_stats = {"llm_calls": 0, "candidate_evaluations": 0, "retries": 0}

        def embed(self, *, characters, positions, progress_callback, **kwargs):
            progress_callback(0, len(characters), "embedding")
            self._embedding_stats["retries"] = 1
            progress_callback(0, len(characters), "retry 1/12")
            raise RuntimeError("candidate generation failed")

    monkeypatch.setattr(generator_module, "LLMGenerator", FakeGenerator)
    monkeypatch.setattr(embedder_module, "EmbedderLLM", FailedEmbedder)
    emitted = []

    with pytest.raises(DemoPipelineError, match="Real LLM embedding failed"):
        run_sender(
            message="hello",
            password="correct horse",
            topic="A quiet garden",
            on_stage=lambda name, data: emitted.append((name, data)),
        )

    updates = [data for name, data in emitted if name == "embedding"]
    assert updates[-1]["current_stage"] == "Retrying candidate generation"
    assert "Retry 1/12 for character 1/" in updates[-1]["message"]
    assert all(update.get("completed_characters", 0) == 0 for update in updates)
    assert not any(data.get("status") == "success" for data in updates)


def test_sender_logs_embedding_traceback_without_exposing_details_to_caller(
    monkeypatch, caplog
):
    from app.llm import embedder as embedder_module
    from app.llm import generator as generator_module

    class FakeGenerator:
        model_name = "test-model"

    class InvalidEmbedding(FakeGenerator):
        def __init__(self, llm_generator):
            self._embedding_stats = {
                "llm_calls": 0,
                "candidate_evaluations": 0,
                "retries": 0,
            }

        def embed(self, **kwargs):
            raise ValueError("specific invalid embedding condition")

    monkeypatch.setattr(generator_module, "LLMGenerator", FakeGenerator)
    monkeypatch.setattr(embedder_module, "EmbedderLLM", InvalidEmbedding)
    caplog.set_level("ERROR", logger="app.web_demo.embedding")

    with pytest.raises(DemoPipelineError) as raised:
        run_sender(
            message="never logged",
            password="never logged",
            topic="A quiet garden",
        )

    assert str(raised.value) == (
        "Real LLM embedding failed (ValueError). "
        "Check model availability and server output."
    )
    assert "specific invalid embedding condition" in caplog.text
    assert "Traceback (most recent call last)" in caplog.text
    assert "never logged" not in caplog.text


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
