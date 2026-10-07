import socket
import threading
import time
from types import SimpleNamespace

import pytest

from server_a import send_message
from server_b import receive_message


class _FakeGenerator:
    model_name = "test-model"


class _FakeEmbedder:
    position_shift = 0

    def __init__(self, llm_generator, character_map):
        self.llm_generator = llm_generator
        self.character_map = character_map

    def embed(self, topic, characters, positions, initial_story, **kwargs):
        positions = [position + self.position_shift for position in positions]
        story = list(initial_story)
        if positions and len(story) <= positions[-1]:
            story.extend(" " * (positions[-1] + 1 - len(story)))
        for character, position in zip(characters, positions):
            story[position] = character
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
def test_two_server_pipeline_round_trip_hi(mode, position_shift, monkeypatch, capsys):
    from app.llm import embedder, generator

    monkeypatch.setattr(embedder, "EmbedderLLM", _FakeEmbedder)
    monkeypatch.setattr(generator, "LLMGenerator", _FakeGenerator)
    monkeypatch.setattr(_FakeEmbedder, "position_shift", position_shift)
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
        "HI",
        "A quiet city street at dusk",
        mode=mode,
        password="test-password" if mode == "psk" else None,
    )
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert not errors
    output = capsys.readouterr().out
    if mode == "ecdhe":
        assert "Shared secrets match   : PASS" in output
        assert "TCP public-key exchange: PASS" in output
    else:
        assert "Password-based key establishment" in output
    assert "Mapped payload length    : 36 characters" in output
    if position_shift:
        assert "Adaptive target positions: authenticated" in output
    assert "Recovered message        : HI" in output
    assert "Message match            : PASS" in output
    assert "Overall result           : PASS" in output
