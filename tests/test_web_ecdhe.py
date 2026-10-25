from types import SimpleNamespace

import pytest

from app import web_demo
from app.crypto.mapping import CharacterMap
from app.tcp_protocol import payload_associated_data
from app.web_demo import DemoPipelineError, create_ecdhe_session, run_receiver, run_sender


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

        def embed(self, *, characters, positions, initial_story="", **kwargs):
            progress_callback = kwargs.get("progress_callback")
            story = list(initial_story)
            if positions and len(story) <= positions[-1]:
                story.extend(" " * (positions[-1] + 1 - len(story)))
            if progress_callback is not None:
                progress_callback(0, len(characters), "embedding")
            for index, (character, position) in enumerate(zip(characters, positions)):
                story[position] = character
                if progress_callback is not None:
                    progress_callback(index + 1, len(characters), "embedded")
            return SimpleNamespace(
                story="".join(story),
                positions=list(positions),
                attempts=1,
            )

        @staticmethod
        def verify_embedding(story, characters, positions):
            return all(
                story[position].upper() == character.upper()
                for character, position in zip(characters, positions)
            )

    monkeypatch.setattr(generator_module, "LLMGenerator", FakeGenerator)
    monkeypatch.setattr(embedder_module, "EmbedderLLM", FakeEmbedder)


def _send_ecdhe_message(handshake, *, message="hello"):
    return run_sender(
        message=message,
        topic="A quiet garden",
        mode="ECDHE",
        ecdhe_session_id=handshake["session_id"],
    )


def _receive_ecdhe_message(sender, *, carrier_text=None, **changes):
    metadata = {
        "carrier_text": (
            sender["stages"]["carrier"]["text"]
            if carrier_text is None
            else carrier_text
        ),
        "salt_hex": sender["session"]["salt_hex"],
        "nonce_hex": sender["session"]["nonce_hex"],
        "mapped_length": sender["session"]["mapped_length"],
        "mode": "ECDHE",
        "ecdhe_session_id": sender["session"]["ecdhe_session_id"],
    }
    metadata.update(changes)
    return run_receiver(**metadata)


def test_handshake_independently_derives_matching_shared_and_message_keys(monkeypatch):
    shared_secrets = []
    derivations = []
    real_shared_secret = web_demo.derive_shared_secret
    real_derive_keys = web_demo.derive_ecdhe_keys

    def capture_shared_secret(private_key, peer_public_key):
        result = real_shared_secret(private_key, peer_public_key)
        shared_secrets.append(result)
        return result

    def capture_derived_keys(shared_secret, salt):
        result = real_derive_keys(shared_secret, salt)
        derivations.append((bytes(shared_secret), bytes(salt), result))
        return result

    monkeypatch.setattr(web_demo, "derive_shared_secret", capture_shared_secret)
    monkeypatch.setattr(web_demo, "derive_ecdhe_keys", capture_derived_keys)

    events = []
    handshake = create_ecdhe_session(
        on_stage=lambda name, data: events.append((name, data))
    )

    assert len(shared_secrets) == 2
    assert shared_secrets[0] == shared_secrets[1]
    assert len(derivations) == 2
    assert derivations[0][0] == derivations[1][0] == shared_secrets[0]
    assert derivations[0][1] == derivations[1][1]
    assert derivations[0][2] == derivations[1][2]
    assert [name for name, data in events if data["status"] == "success"] == [
        "ecdhe_key_pair_generation",
        "ecdhe_public_key_exchange",
        "ecdhe_shared_secret",
        "ecdhe_key_derivation",
    ]
    assert handshake["mode"] == "ECDHE"
    assert "private" not in str(handshake).lower()
    assert "shared_secret" not in str(handshake).lower()
    assert "dk1" not in str(handshake).lower()


def test_ecdhe_uses_shared_message_pipeline_and_recovers_plaintext(fake_embedder):
    handshake = create_ecdhe_session()
    sender = _send_ecdhe_message(handshake)

    assert sender["mode"] == "ECDHE"
    assert sender["stages"]["encryption"]["algorithm"] == "AES-256-GCM"
    assert sender["stages"]["payload"]["order"] == "tag || ciphertext"
    assert sender["stages"]["positions"]["algorithm"] == "SHAKE-128"
    assert sender["stages"]["encryption"]["aad_hex"] == (
        payload_associated_data(sender["session"]["mapped_length"]).hex()
    )
    assert "dk1" not in sender["stages"]["pbkdf2"]
    assert "dk2" not in sender["stages"]["pbkdf2"]

    receiver_events = []
    receiver = run_receiver(
        carrier_text=sender["stages"]["carrier"]["text"],
        salt_hex=sender["session"]["salt_hex"],
        nonce_hex=sender["session"]["nonce_hex"],
        mapped_length=sender["session"]["mapped_length"],
        mode="ECDHE",
        ecdhe_session_id=sender["session"]["ecdhe_session_id"],
        on_stage=lambda name, data: receiver_events.append((name, data)),
    )

    assert receiver["mode"] == "ECDHE"
    assert receiver["stages"]["authentication"]["status"] == "success"
    assert receiver["stages"]["authentication"]["plaintext"] == "hello"
    assert receiver["stages"]["key_derivation"]["key_source"] == "ECDHE shared secret"
    assert [name for name, data in receiver_events if data["status"] == "success"] == [
        "key_derivation",
        "extraction",
        "inverse_mapping",
        "authentication",
    ]


def test_ecdhe_rejects_incomplete_wrong_and_mismatched_sessions(fake_embedder):
    handshake = create_ecdhe_session()

    with pytest.raises(DemoPipelineError, match="does not accept a password"):
        run_sender(
            message="hello",
            password="must not become an ECDHE secret",
            topic="A quiet garden",
            mode="ECDHE",
            ecdhe_session_id=handshake["session_id"],
        )

    with pytest.raises(DemoPipelineError, match="incomplete"):
        run_receiver(
            carrier_text="not a carrier",
            salt_hex="00" * 16,
            nonce_hex="00" * 12,
            mapped_length=36,
            mode="ECDHE",
            ecdhe_session_id=handshake["session_id"],
        )

    with pytest.raises(DemoPipelineError, match="missing or expired"):
        run_sender(
            message="hello",
            topic="A quiet garden",
            mode="ECDHE",
            ecdhe_session_id="not-a-valid-session",
        )

    sender = _send_ecdhe_message(handshake)
    with pytest.raises(DemoPipelineError, match="missing or expired"):
        _receive_ecdhe_message(sender, ecdhe_session_id="another-session")
    with pytest.raises(DemoPipelineError, match="metadata does not match"):
        _receive_ecdhe_message(sender, nonce_hex="00" * 12)
    with pytest.raises(DemoPipelineError, match="already been used"):
        _send_ecdhe_message(handshake)


def test_ecdhe_modified_carrier_fails_authentication_without_plaintext(fake_embedder):
    handshake = create_ecdhe_session()
    sender = _send_ecdhe_message(handshake)
    carrier = list(sender["stages"]["carrier"]["text"])
    first_position = sender["stages"]["positions"]["positions"][0]
    original = carrier[first_position]
    carrier[first_position] = next(
        character for character in CharacterMap.ALPHABET if character != original
    )

    receiver = _receive_ecdhe_message(sender, carrier_text="".join(carrier))

    assert receiver["stages"]["authentication"]["status"] == "failed"
    assert "plaintext" not in receiver["stages"]["authentication"]


def test_ecdhe_handshake_failure_is_reported_without_a_session(monkeypatch):
    events = []

    def fail_key_generation():
        raise RuntimeError("test-only failure")

    monkeypatch.setattr(web_demo, "generate_key_pair", fail_key_generation)

    with pytest.raises(DemoPipelineError, match="key-pair generation failed"):
        create_ecdhe_session(on_stage=lambda name, data: events.append((name, data)))

    assert events[-1] == (
        "ecdhe_key_pair_generation",
        {"status": "failed", "message": "X25519 key-pair generation failed."},
    )
