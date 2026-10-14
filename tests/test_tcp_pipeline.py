import socket
import threading
import time
from types import SimpleNamespace

import pytest

from server_a import send_message
from server_b import receive_message


class _FakeGenerator:
    model_name = "test-model"
    instances = 0

    def __init__(self):
        type(self).instances += 1


class _FakeEmbedder:
    position_shift = 0
    requested_positions = []
    actual_positions = []
    embedded_lengths = []

    def __init__(self, llm_generator, character_map):
        self.llm_generator = llm_generator
        self.character_map = character_map

    def embed(self, topic, characters, positions, initial_story, **kwargs):
        type(self).embedded_lengths.append(len(characters))
        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None:
            progress_callback(0, len(characters), "embedding")
        type(self).requested_positions = list(positions)
        position_shift = (
            self.position_shift
            if kwargs.get("allow_adaptive_positions", False)
            else 0
        )
        positions = [position + position_shift for position in positions]
        type(self).actual_positions = list(positions)
        story = list(initial_story)
        if positions and len(story) <= positions[-1]:
            story.extend(" " * (positions[-1] + 1 - len(story)))
        for character, position in zip(characters, positions):
            story[position] = character
        if progress_callback is not None:
            progress_callback(len(characters), len(characters), "embedded")
        return SimpleNamespace(
            story="".join(story),
            positions=list(positions),
            attempts=len(characters),
        )

    @staticmethod
    def verify_embedding(story, characters, positions):
        return all(
            story[position].upper() == character.upper()
            for character, position in zip(characters, positions)
        )

    @staticmethod
    def _validate_cover_naturalness(story, topic):
        return {"test_cover": True}


@pytest.mark.parametrize("mode", ["psk", "ecdhe"])
@pytest.mark.parametrize("position_shift", [0, 1])
@pytest.mark.parametrize("secret", ["HI", "hi"])
def test_two_server_pipeline_round_trip_hi(mode, position_shift, secret, monkeypatch, capsys):
    from app.llm import embedder, generator

    monkeypatch.setattr(embedder, "EmbedderLLM", _FakeEmbedder)
    monkeypatch.setattr(generator, "LLMGenerator", _FakeGenerator)
    monkeypatch.setattr(_FakeEmbedder, "position_shift", position_shift)
    _FakeEmbedder.embedded_lengths = []
    monkeypatch.setattr("server_b.getpass.getpass", lambda prompt: "test-password")

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    errors = []

    def receiver():
        try:
            receive_message("127.0.0.1", port)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=receiver)
    thread.start()
    time.sleep(0.05)

    send_message(
        "127.0.0.1",
        port,
        secret,
        "A quiet city street at dusk",
        mode=mode,
        password="test-password" if mode == "psk" else None,
    )
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert not errors
    assert len(_FakeEmbedder.requested_positions) == 36
    if mode == "ecdhe":
        assert _FakeEmbedder.embedded_lengths == [
            28, 28, 28, 28, 24, 24, 24, 24, 24, 24, 36
        ]
    else:
        assert _FakeEmbedder.embedded_lengths == [36]
    expected_positions = [
        position + position_shift
        for position in _FakeEmbedder.requested_positions
    ]
    assert _FakeEmbedder.actual_positions == expected_positions
    if not position_shift:
        assert _FakeEmbedder.actual_positions == _FakeEmbedder.requested_positions
    output = capsys.readouterr().out
    if mode == "ecdhe":
        assert "ECDHE key exchange complete (10 alternating carriers)" in output
        assert "Carrier 1/10 sent (Server A)" in output
        assert "Carrier 2/10 received (Server B)" in output
    assert "LLM embedding: 36/36 characters (embedded)" in output
    assert f"{mode.upper()} carrier sent to Server B." in output
    assert f"Received and authenticated message: {secret}" in output
    assert "AES-256-GCM authentication: PASS" in output
    assert "Embedding 1/" not in output
    assert "Character stats:" not in output


def test_ecdhe_requests_secret_only_after_public_key_carriers_and_key_confirmation(
    monkeypatch, capsys
):
    import builtins

    from app.llm import embedder, generator

    monkeypatch.setattr(embedder, "EmbedderLLM", _FakeEmbedder)
    monkeypatch.setattr(generator, "LLMGenerator", _FakeGenerator)
    monkeypatch.setattr(_FakeEmbedder, "position_shift", 0)
    _FakeEmbedder.embedded_lengths = []

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    errors = []

    def receiver():
        try:
            receive_message("127.0.0.1", port)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=receiver)
    thread.start()
    time.sleep(0.05)

    prompt_output = []

    def enter_secret(prompt):
        prompt_output.append(capsys.readouterr().out)
        return "HI"

    monkeypatch.setattr(builtins, "input", enter_secret)
    send_message(
        "127.0.0.1",
        port,
        "",
        "A quiet city street at dusk",
        mode="ecdhe",
    )
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert not errors
    assert len(_FakeEmbedder.embedded_lengths) == 11
    assert len(prompt_output) == 1
    assert "ECDHE key exchange complete (10 alternating carriers)" in prompt_output[0]
    assert "Received and authenticated message: HI" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["psk", "ecdhe"])
def test_demo_cli_connects_to_external_receiver(mode, monkeypatch, capsys):
    from app import demo
    import server_a
    import server_b

    forwarded = {}
    monkeypatch.setattr(server_a, "send_message", lambda *args, **kwargs: forwarded.update({
        "args": args,
        **kwargs,
    }))
    monkeypatch.setattr(
        server_b,
        "receive_message",
        lambda *args, **kwargs: pytest.fail("app.demo must not start Server B"),
    )
    monkeypatch.setattr(demo.getpass, "getpass", lambda prompt: "do-not-print-this")
    command = [
        "--mode",
        mode,
        "--topic",
        "A quiet city street at dusk",
        "--secret",
        "HI",
    ]
    demo.main(command)

    assert forwarded["args"] == (
        "127.0.0.1",
        50505,
        "HI",
        "A quiet city street at dusk",
    )
    assert forwarded["mode"] == mode
    assert forwarded["demonstration"] is True
    assert forwarded["password"] == ("do-not-print-this" if mode == "psk" else None)
    assert "do-not-print-this" not in capsys.readouterr().out
