"""Backend orchestration for the local PSK demonstration UI."""

from __future__ import annotations

from typing import Any

from app.config import DEFAULT_PBKDF2_CONFIG, get_pbkdf2_iterations, get_model_name
from app.crypto.aead import AEADError, decrypt, encrypt
from app.crypto.aead_mapping import (
    aead_to_character_sequence,
    character_sequence_to_aead,
)
from app.crypto.key_derivation import derive_keys, generate_salt
from app.extraction.extractor import Extractor
from app.tcp_protocol import (
    MAX_MAPPED_CHARACTERS,
    payload_associated_data,
    positions_for_payload,
)


class DemoPipelineError(ValueError):
    """A safe-to-display error associated with a specific pipeline stage."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def _require_text(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise DemoPipelineError("input", f"{name} must be text.")
    if not allow_empty and not value.strip():
        raise DemoPipelineError("input", f"{name} must not be empty.")
    return value


def _embedding_rows(mapped: str, positions: list[int]) -> list[dict[str, Any]]:
    return [
        {"index": index + 1, "character": character, "position": position}
        for index, (character, position) in enumerate(zip(mapped, positions))
    ]


def run_sender(
    *,
    message: object,
    password: object,
    topic: object,
    temperature: object = 0.7,
    top_k: object = 20,
) -> dict[str, Any]:
    """Run the real PSK sender, embedder, extraction, and verification path."""

    message = _require_text(message, "Secret message")
    password = _require_text(password, "Password")
    topic = _require_text(topic, "Carrier topic")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise DemoPipelineError("input", "Temperature must be a number.")
    if not 0 < temperature <= 0.9:
        raise DemoPipelineError("input", "Temperature must be greater than 0 and at most 0.9.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 100:
        raise DemoPipelineError("input", "Top-k must be an integer from 1 to 100.")

    plaintext = message.encode("utf-8")
    mapped_length = (len(plaintext) + 16) * 2
    if mapped_length > MAX_MAPPED_CHARACTERS:
        raise DemoPipelineError(
            "input",
            f"Message is too large for the configured {MAX_MAPPED_CHARACTERS}-character payload limit.",
        )

    try:
        salt = generate_salt()
        dk1, dk2 = derive_keys(password, salt)
        associated_data = payload_associated_data(mapped_length)
        encrypted = encrypt(plaintext, dk1, associated_data)
        mapped = aead_to_character_sequence(encrypted["enc"])
        positions = positions_for_payload(dk2, len(mapped))
    except (TypeError, ValueError, OverflowError) as error:
        raise DemoPipelineError("cryptography", "Key derivation or encryption failed.") from error

    try:
        from server_a import _safe_initial_story
        from app.llm.embedder import EmbedderLLM
        from app.llm.generator import LLMGenerator

        generator = LLMGenerator()
        embedder = EmbedderLLM(llm_generator=generator)
        result = embedder.embed(
            topic=topic,
            characters=mapped,
            positions=positions,
            initial_story=_safe_initial_story(topic, positions[0]),
            temperature=float(temperature),
            top_k=top_k,
            max_new_tokens=32,
            max_attempts=15000,
            max_retries=12,
            allow_adaptive_positions=False,
            verbose=False,
        )
    except (ImportError, OSError, RuntimeError, ValueError, MemoryError) as error:
        raise DemoPipelineError(
            "embedding",
            f"Real LLM embedding failed ({type(error).__name__}). Check model availability and server output.",
        ) from error

    actual_positions = result.positions
    try:
        embedded_ok = EmbedderLLM.verify_embedding(
            result.story,
            mapped,
            actual_positions,
        )
        extracted = Extractor().extract(result.story, actual_positions).upper()
        recovered_enc = character_sequence_to_aead(extracted)
        recovered_plaintext = decrypt(
            ciphertext=recovered_enc[16:],
            tag=recovered_enc[:16],
            nonce=encrypted["nonce"],
            dk1=dk1,
            associated_data=associated_data,
        )
    except (AEADError, IndexError, TypeError, ValueError) as error:
        raise DemoPipelineError(
            "verification",
            f"Sender-side extraction or authenticated verification failed ({type(error).__name__}).",
        ) from error

    if not embedded_ok or extracted != mapped or recovered_enc != encrypted["enc"]:
        raise DemoPipelineError(
            "verification",
            "Sender-side embedding, extraction, and payload verification did not match.",
        )
    if recovered_plaintext != plaintext:
        raise DemoPipelineError("verification", "Recovered bytes did not match the original message.")

    stats = getattr(embedder, "_embedding_stats", {})
    return {
        "mode": "PSK",
        "model": str(generator.model_name),
        "session": {
            "salt_hex": salt.hex(),
            "nonce_hex": encrypted["nonce"].hex(),
            "mapped_length": len(mapped),
        },
        "stages": {
            "key_establishment": {
                "status": "success",
                "mode": "PSK",
                "explanation": (
                    "The sender derives keys from the entered pre-shared password. "
                    "The receiver independently derives matching keys; successful AEAD authentication verifies them."
                ),
            },
            "pbkdf2": {
                "status": "success",
                "algorithm": f"PBKDF2-HMAC-{DEFAULT_PBKDF2_CONFIG.hash_name.upper()}",
                "iterations": get_pbkdf2_iterations(),
                "salt_hex": salt.hex(),
                "derived_key_length": DEFAULT_PBKDF2_CONFIG.key_length,
                "dk1_length": len(dk1),
                "dk2_length": len(dk2),
                "explanation": "DK1 is the AES-256-GCM key; DK2 regenerates SHAKE-128 positions.",
            },
            "encryption": {
                "status": "success",
                "plaintext": message,
                "algorithm": "AES-256-GCM",
                "nonce_hex": encrypted["nonce"].hex(),
                "aad_hex": associated_data.hex(),
                "ciphertext_hex": encrypted["ciphertext"].hex(),
                "tag_hex": encrypted["tag"].hex(),
                "ciphertext_length": len(encrypted["ciphertext"]),
                "tag_length": len(encrypted["tag"]),
            },
            "payload": {
                "status": "success",
                "order": "tag || ciphertext",
                "hex": encrypted["enc"].hex(),
                "byte_length": len(encrypted["enc"]),
                "hex_character_length": len(encrypted["enc"].hex()),
            },
            "mapping": {
                "status": "success",
                "input_hex": encrypted["enc"].hex(),
                "output": mapped,
                "input_byte_length": len(encrypted["enc"]),
                "output_character_length": len(mapped),
                "explanation": "The project maps each payload byte to two h4 alphabet characters.",
            },
            "positions": {
                "status": "success",
                "algorithm": "SHAKE-128",
                "count": len(positions),
                "positions": actual_positions,
                "rows": _embedding_rows(mapped, actual_positions),
                "key_source": "DK2 (not displayed)",
            },
            "embedding": {
                "status": "success",
                "topic": topic,
                "model": str(generator.model_name),
                "temperature": float(temperature),
                "top_k": top_k,
                "mapped_character_count": len(mapped),
                "target_positions": actual_positions,
                "attempts": result.attempts,
                "llm_calls": stats.get("llm_calls"),
                "candidate_evaluations": stats.get("candidate_evaluations"),
                "retries": stats.get("retries"),
            },
            "carrier": {
                "status": "success",
                "text": result.story,
                "length": len(result.story),
                "verified": embedded_ok,
                "notice": (
                    "Carrier text is intended to conceal an encrypted payload in generated text; "
                    "this does not guarantee undetectability."
                ),
            },
            "sender_verification": {
                "status": "success",
                "extracted": extracted,
                "authentication": "success",
                "recovered_plaintext": recovered_plaintext.decode("utf-8"),
            },
        },
    }


def run_receiver(
    *,
    carrier_text: object,
    password: object,
    salt_hex: object,
    nonce_hex: object,
    mapped_length: object,
) -> dict[str, Any]:
    """Regenerate PSK positions, extract the payload, and authenticate it."""

    carrier_text = _require_text(carrier_text, "Received carrier text")
    password = _require_text(password, "Password")
    salt_hex = _require_text(salt_hex, "Salt")
    nonce_hex = _require_text(nonce_hex, "Nonce")
    if (
        not isinstance(mapped_length, int)
        or isinstance(mapped_length, bool)
        or mapped_length <= 0
        or mapped_length % 2
        or mapped_length > MAX_MAPPED_CHARACTERS
    ):
        raise DemoPipelineError("input", "Mapped payload length is invalid.")
    try:
        salt = bytes.fromhex(salt_hex)
        nonce = bytes.fromhex(nonce_hex)
        if not salt:
            raise ValueError("empty salt")
        if len(nonce) != 12:
            raise ValueError("invalid nonce length")
    except ValueError as error:
        raise DemoPipelineError("input", "Salt or nonce metadata is invalid.") from error

    try:
        dk1, dk2 = derive_keys(password, salt)
        associated_data = payload_associated_data(mapped_length)
        positions = positions_for_payload(dk2, mapped_length)
        extracted = Extractor.extract(carrier_text, positions).upper()
        if len(extracted) != mapped_length:
            raise DemoPipelineError(
                "extraction",
                "Carrier text is too short or does not contain every expected target character.",
            )
    except DemoPipelineError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise DemoPipelineError("extraction", "Position regeneration or extraction failed.") from error

    try:
        recovered_enc = character_sequence_to_aead(extracted)
        if len(recovered_enc) < 16:
            raise ValueError("payload is shorter than an authentication tag")
    except (TypeError, ValueError) as error:
        raise DemoPipelineError("inverse_mapping", "Extracted characters are not a valid h4 AEAD payload.") from error

    ciphertext = recovered_enc[16:]
    tag = recovered_enc[:16]
    try:
        plaintext = decrypt(
            ciphertext=ciphertext,
            tag=tag,
            nonce=nonce,
            dk1=dk1,
            associated_data=associated_data,
        )
    except AEADError:
        return {
            "mode": "PSK",
            "stages": {
                "key_derivation": {
                    "status": "success",
                    "algorithm": f"PBKDF2-HMAC-{DEFAULT_PBKDF2_CONFIG.hash_name.upper()}",
                    "iterations": get_pbkdf2_iterations(),
                    "dk1_length": len(dk1),
                    "dk2_length": len(dk2),
                },
                "extraction": {
                    "status": "success",
                    "positions": positions,
                    "extracted": extracted,
                },
                "inverse_mapping": {
                    "status": "success",
                    "payload_hex": recovered_enc.hex(),
                    "tag_hex": tag.hex(),
                    "ciphertext_hex": ciphertext.hex(),
                    "tag_length": len(tag),
                    "ciphertext_length": len(ciphertext),
                },
                "authentication": {
                    "status": "failed",
                    "message": "AES-256-GCM authentication failed. No plaintext was recovered.",
                },
            },
        }

    return {
        "mode": "PSK",
        "stages": {
            "key_derivation": {
                "status": "success",
                "algorithm": f"PBKDF2-HMAC-{DEFAULT_PBKDF2_CONFIG.hash_name.upper()}",
                "iterations": get_pbkdf2_iterations(),
                "dk1_length": len(dk1),
                "dk2_length": len(dk2),
            },
            "extraction": {
                "status": "success",
                "positions": positions,
                "extracted": extracted,
            },
            "inverse_mapping": {
                "status": "success",
                "payload_hex": recovered_enc.hex(),
                "tag_hex": tag.hex(),
                "ciphertext_hex": ciphertext.hex(),
                "tag_length": len(tag),
                "ciphertext_length": len(ciphertext),
            },
            "authentication": {
                "status": "success",
                "message": "AES-256-GCM authentication succeeded.",
                "plaintext": plaintext.decode("utf-8"),
            },
        },
    }


def get_demo_config() -> dict[str, Any]:
    """Return non-secret values needed to render the demo settings."""

    return {
        "mode": "PSK",
        "model": get_model_name(),
        "pbkdf2_algorithm": f"PBKDF2-HMAC-{DEFAULT_PBKDF2_CONFIG.hash_name.upper()}",
        "pbkdf2_iterations": get_pbkdf2_iterations(),
        "dk_length": DEFAULT_PBKDF2_CONFIG.key_length,
        "max_mapped_characters": MAX_MAPPED_CHARACTERS,
    }
