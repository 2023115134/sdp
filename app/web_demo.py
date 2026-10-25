"""Backend orchestration for the local PSK demonstration UI."""

from __future__ import annotations

import hmac
import logging
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from app.config import DEFAULT_PBKDF2_CONFIG, get_pbkdf2_iterations, get_model_name
from app.crypto.aead import AEADError, decrypt, encrypt
from app.crypto.aead_mapping import (
    aead_to_character_sequence,
    character_sequence_to_aead,
)
from app.crypto.ecdhe import derive_ecdhe_keys, derive_shared_secret, generate_key_pair
from app.crypto.key_derivation import derive_keys, generate_salt
from app.extraction.extractor import Extractor
from app.tcp_protocol import (
    MAX_MAPPED_CHARACTERS,
    payload_associated_data,
    positions_for_payload,
)

embedding_logger = logging.getLogger("app.web_demo.embedding")
_embedding_terminal_lock = threading.RLock()
_embedding_terminal_state = threading.local()
_transformers_tqdm_lock = threading.RLock()


class DemoPipelineError(ValueError):
    """A safe-to-display error associated with a specific pipeline stage."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


@dataclass
class _ECDHESession:
    salt: bytes
    sender_dk1: bytes
    sender_dk2: bytes
    receiver_dk1: bytes
    receiver_dk2: bytes
    created_at: float
    status: str = "ready"
    nonce_hex: str | None = None
    mapped_length: int | None = None


_ECDHE_SESSION_TTL_SECONDS = 900
_ECDHE_MAX_ACTIVE_SESSIONS = 128
_ecdhe_sessions: dict[str, _ECDHESession] = {}
_ecdhe_sessions_lock = threading.Lock()


def _emit_stage(
    callback: Callable[[str, dict[str, Any]], None] | None,
    name: str,
    data: dict[str, Any],
) -> None:
    if callback is not None:
        callback(name, data)


def create_ecdhe_session(
    *,
    on_stage: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Perform a real, in-process X25519 handshake for two logical endpoints."""

    stage_name = "ecdhe_key_pair_generation"
    _emit_stage(
        on_stage,
        stage_name,
        {"status": "running", "message": "Generating sender endpoint X25519 key pair."},
    )
    try:
        sender_private, sender_public = generate_key_pair()
        _emit_stage(
            on_stage,
            stage_name,
            {
                "status": "running",
                "message": "Sender key pair generated; generating receiver endpoint key pair.",
            },
        )
        receiver_private, receiver_public = generate_key_pair()
    except (TypeError, ValueError, RuntimeError) as error:
        _emit_stage(
            on_stage,
            stage_name,
            {"status": "failed", "message": "X25519 key-pair generation failed."},
        )
        raise DemoPipelineError(
            "key_establishment", "ECDHE key-pair generation failed."
        ) from error
    _emit_stage(
        on_stage,
        stage_name,
        {
            "status": "success",
            "algorithm": "X25519",
            "message": "Both backend-only endpoint key pairs were generated.",
        },
    )

    stage_name = "ecdhe_public_key_exchange"
    _emit_stage(
        on_stage,
        stage_name,
        {"status": "running", "message": "Exchanging public keys between local endpoints."},
    )
    sender_peer_public = receiver_public
    receiver_peer_public = sender_public
    _emit_stage(
        on_stage,
        stage_name,
        {
            "status": "success",
            "transport": "in-process local demonstration",
            "message": "Each endpoint received the other endpoint's public key.",
        },
    )

    stage_name = "ecdhe_shared_secret"
    _emit_stage(
        on_stage,
        stage_name,
        {"status": "running", "message": "Each endpoint is calculating its X25519 shared secret."},
    )
    try:
        sender_shared_secret = derive_shared_secret(sender_private, sender_peer_public)
        receiver_shared_secret = derive_shared_secret(receiver_private, receiver_peer_public)
    except (TypeError, ValueError, RuntimeError) as error:
        _emit_stage(
            on_stage,
            stage_name,
            {"status": "failed", "message": "X25519 shared-secret calculation failed."},
        )
        raise DemoPipelineError(
            "key_establishment", "ECDHE shared-secret calculation failed."
        ) from error

    if not hmac.compare_digest(sender_shared_secret, receiver_shared_secret):
        _emit_stage(
            on_stage,
            stage_name,
            {"status": "failed", "message": "Endpoint shared secrets did not match."},
        )
        raise DemoPipelineError(
            "key_establishment", "ECDHE endpoints did not derive the same shared secret."
        )
    _emit_stage(
        on_stage,
        stage_name,
        {
            "status": "success",
            "match": True,
            "message": "The independently calculated endpoint secrets match; secret bytes are not exposed.",
        },
    )

    stage_name = "ecdhe_key_derivation"
    _emit_stage(
        on_stage,
        stage_name,
        {"status": "running", "message": "Deriving each endpoint's DK1 and DK2 with PBKDF2."},
    )
    try:
        salt = generate_salt()
        sender_dk1, sender_dk2 = derive_ecdhe_keys(sender_shared_secret, salt)
        receiver_dk1, receiver_dk2 = derive_ecdhe_keys(receiver_shared_secret, salt)
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        _emit_stage(
            on_stage,
            stage_name,
            {"status": "failed", "message": "ECDHE message-key derivation failed."},
        )
        raise DemoPipelineError(
            "key_establishment", "ECDHE message-key derivation failed."
        ) from error

    if not (
        hmac.compare_digest(sender_dk1, receiver_dk1)
        and hmac.compare_digest(sender_dk2, receiver_dk2)
    ):
        _emit_stage(
            on_stage,
            stage_name,
            {"status": "failed", "message": "Endpoint message keys did not match."},
        )
        raise DemoPipelineError(
            "key_establishment", "ECDHE endpoints derived incompatible message keys."
        )

    session_id = secrets.token_urlsafe(32)
    now = time.monotonic()
    session_limit_reached = False
    with _ecdhe_sessions_lock:
        expired = [
            key
            for key, session in _ecdhe_sessions.items()
            if now - session.created_at >= _ECDHE_SESSION_TTL_SECONDS
        ]
        for key in expired:
            del _ecdhe_sessions[key]
        if len(_ecdhe_sessions) >= _ECDHE_MAX_ACTIVE_SESSIONS:
            session_limit_reached = True
        else:
            _ecdhe_sessions[session_id] = _ECDHESession(
                salt=salt,
                sender_dk1=sender_dk1,
                sender_dk2=sender_dk2,
                receiver_dk1=receiver_dk1,
                receiver_dk2=receiver_dk2,
                created_at=now,
            )
    if session_limit_reached:
        _emit_stage(
            on_stage,
            stage_name,
            {"status": "failed", "message": "The local ECDHE session limit was reached."},
        )
        raise DemoPipelineError(
            "key_establishment", "Too many active ECDHE sessions; wait for one to expire."
        )
    _emit_stage(
        on_stage,
        stage_name,
        {
            "status": "success",
            "algorithm": f"PBKDF2-HMAC-{DEFAULT_PBKDF2_CONFIG.hash_name.upper()}",
            "iterations": get_pbkdf2_iterations(),
            "salt_hex": salt.hex(),
            "dk1_length": len(sender_dk1),
            "dk2_length": len(sender_dk2),
            "message": "Both endpoints independently derived matching DK1/DK2; key bytes remain backend-only.",
        },
    )
    return {
        "mode": "ECDHE",
        "session_id": session_id,
        "demonstration": "single-computer logical endpoints; unauthenticated public-key exchange",
    }


def _claim_ecdhe_sender_session(session_id: object) -> tuple[str, _ECDHESession]:
    if not isinstance(session_id, str) or not session_id:
        raise DemoPipelineError(
            "key_establishment", "A completed ECDHE handshake is required."
        )
    with _ecdhe_sessions_lock:
        session = _ecdhe_sessions.get(session_id)
        if session is None or time.monotonic() - session.created_at >= _ECDHE_SESSION_TTL_SECONDS:
            _ecdhe_sessions.pop(session_id, None)
            raise DemoPipelineError(
                "key_establishment", "ECDHE session is missing or expired; establish a new handshake."
            )
        if session.status != "ready":
            raise DemoPipelineError(
                "key_establishment", "ECDHE session is incomplete or has already been used."
            )
        session.status = "sender_running"
        return session_id, session


def _complete_ecdhe_sender_session(
    session_id: str,
    session: _ECDHESession,
    *,
    nonce_hex: str,
    mapped_length: int,
) -> None:
    with _ecdhe_sessions_lock:
        if _ecdhe_sessions.get(session_id) is not session or session.status != "sender_running":
            raise DemoPipelineError(
                "key_establishment", "ECDHE sender session state changed unexpectedly."
            )
        session.nonce_hex = nonce_hex
        session.mapped_length = mapped_length
        session.status = "sender_complete"


def _get_ecdhe_receiver_keys(
    session_id: object,
    *,
    salt_hex: object,
    nonce_hex: object,
    mapped_length: object,
) -> tuple[bytes, bytes, bytes]:
    if not isinstance(session_id, str) or not session_id:
        raise DemoPipelineError(
            "key_derivation", "A completed ECDHE sender session is required."
        )
    with _ecdhe_sessions_lock:
        session = _ecdhe_sessions.get(session_id)
        if session is None or time.monotonic() - session.created_at >= _ECDHE_SESSION_TTL_SECONDS:
            _ecdhe_sessions.pop(session_id, None)
            raise DemoPipelineError(
                "key_derivation", "ECDHE session is missing or expired; establish a new handshake."
            )
        if session.status != "sender_complete":
            raise DemoPipelineError(
                "key_derivation", "ECDHE handshake or sender pipeline is incomplete."
            )
        if (
            salt_hex != session.salt.hex()
            or nonce_hex != session.nonce_hex
            or mapped_length != session.mapped_length
        ):
            raise DemoPipelineError(
                "key_derivation", "ECDHE session metadata does not match the established session."
            )
        return session.salt, session.receiver_dk1, session.receiver_dk2


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


def _log_embedding_progress(event: dict[str, object]) -> None:
    """Print actual embedding events as bare terminal lines."""

    event_type = event.get("event")
    index = event.get("index")
    total = event.get("total")
    character = event.get("character")
    position = event.get("position")
    if event_type != "loading_weights":
        _finish_weight_progress_line()
    if event_type == "character_started":
        _embedding_terminal_lock.acquire()
        _embedding_terminal_state.locked = True
        display_character = "SPACE" if character == " " else character
        print(
            f"Embedding {index}/{total}\n"
            f"Character: {display_character}\n"
            f"Position: {position}",
            flush=True,
        )
    elif event_type == "retry_started":
        print(
            "Retry %s: T=%.2f k=%s"
            % (
                event.get("retry_number"),
                event.get("temperature", 0.0),
                event.get("top_k"),
            ),
            flush=True,
        )
    elif event_type == "candidate_rejected":
        print(
            "Candidate rejected: %s (candidate evaluations this retry: %s)"
            % (event.get("reason"), event.get("candidate_evaluations")),
            flush=True,
        )
    elif event_type == "loading_weights":
        current = event.get("current")
        total_weights = event.get("total")
        progress = (
            f"{current}/{total_weights}"
            if total_weights is not None
            else str(current)
        )
        print(f"Loading weights: {progress}", end="\r", flush=True)
        _embedding_terminal_state.weight_progress_active = True
    elif event_type == "character_embedded":
        print(
            "Status: Embedded successfully (candidate evaluations this retry: %s)"
            % event.get("candidate_evaluations"),
            flush=True,
        )
        _release_embedding_terminal_lock()
    elif event_type == "character_failed":
        embedding_logger.error(
            "Status: FAILED to embed %s at position %s after %s retries; "
            "candidate evaluations: %s; reason: %s",
            character,
            position,
            event.get("retry_number"),
            event.get("candidate_evaluations"),
            event.get("reason"),
        )
        _release_embedding_terminal_lock()


def _finish_weight_progress_line() -> None:
    if getattr(_embedding_terminal_state, "weight_progress_active", False):
        print(flush=True)
        _embedding_terminal_state.weight_progress_active = False


def _release_embedding_terminal_lock() -> None:
    if getattr(_embedding_terminal_state, "locked", False):
        _embedding_terminal_state.locked = False
        _embedding_terminal_lock.release()


def _install_transformers_weight_progress_hook(
    callback: Callable[[dict[str, object]], None],
) -> tuple[Callable[..., Any], object] | None:
    try:
        from transformers.utils import logging as transformers_logging
    except ImportError:
        return None

    set_tqdm_hook = getattr(transformers_logging, "set_tqdm_hook", None)
    if not callable(set_tqdm_hook):
        return None

    _transformers_tqdm_lock.acquire()

    def hook(factory: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]):
        if kwargs.get("desc") != "Loading weights":
            return factory(*args, **kwargs)
        if getattr(factory, "__name__", "") == "EmptyTqdm":
            return factory(*args, **kwargs)

        class WeightProgress(factory):
            def __iter__(self):
                if self.disable:
                    yield from super().__iter__()
                    return
                try:
                    for item in self.iterable:
                        yield item
                        self.n += 1
                        self._report_progress()
                finally:
                    self.close()

            def _report_progress(self) -> None:
                if self.n != getattr(self, "_last_reported", -1):
                    self._last_reported = self.n
                    callback(
                        {
                            "event": "loading_weights",
                            "current": self.n,
                            "total": self.total,
                        }
                    )

            def update(self, amount: int = 1):
                result = super().update(amount)
                self._report_progress()
                return result

            def display(self, *display_args: object, **display_kwargs: object) -> None:
                return None

        return WeightProgress(*args, **kwargs)

    try:
        previous_hook = set_tqdm_hook(hook)
    except Exception:
        _transformers_tqdm_lock.release()
        raise
    return set_tqdm_hook, previous_hook


def _restore_transformers_weight_progress_hook(
    hook_state: tuple[Callable[..., Any], object] | None,
) -> None:
    if hook_state is None:
        return
    setter, previous_hook = hook_state
    try:
        setter(previous_hook)
    finally:
        _transformers_tqdm_lock.release()


def run_sender(
    *,
    message: object,
    password: object = None,
    topic: object,
    mode: object = "PSK",
    ecdhe_session_id: object = None,
    temperature: object = 0.7,
    top_k: object = 20,
    on_stage: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the shared PSK or established-ECDHE sender message pipeline."""

    message = _require_text(message, "Secret message")
    topic = _require_text(topic, "Carrier topic")
    if not isinstance(mode, str) or mode not in {"PSK", "ECDHE"}:
        raise DemoPipelineError("input", "Key-establishment mode must be PSK or ECDHE.")
    if mode == "PSK":
        password = _require_text(password, "Password")
    elif password is not None:
        raise DemoPipelineError("input", "ECDHE mode does not accept a password.")
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
        ecdhe_session: _ECDHESession | None = None
        claimed_session_id: str | None = None
        if mode == "PSK":
            salt = generate_salt()
            dk1, dk2 = derive_keys(password, salt)
        else:
            claimed_session_id, ecdhe_session = _claim_ecdhe_sender_session(
                ecdhe_session_id
            )
            salt = ecdhe_session.salt
            dk1, dk2 = ecdhe_session.sender_dk1, ecdhe_session.sender_dk2
        associated_data = payload_associated_data(mapped_length)
        encrypted = encrypt(plaintext, dk1, associated_data)
        mapped = aead_to_character_sequence(encrypted["enc"])
        positions = positions_for_payload(dk2, len(mapped))
    except DemoPipelineError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise DemoPipelineError("cryptography", "Key derivation or encryption failed.") from error

    stages: dict[str, dict[str, Any]] = {
        "key_establishment": {
            "status": "success",
            "mode": mode,
            "explanation": (
                "The sender and receiver use separate DK1/DK2 values derived from their completed backend X25519 handshake. "
                "The local public-key exchange is unauthenticated; this demonstration is single-computer only."
                if mode == "ECDHE"
                else (
                    "The sender derives keys from the entered pre-shared password. "
                    "The receiver independently derives matching keys; successful AEAD authentication verifies them."
                )
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
            "explanation": (
                "PBKDF2 uses the X25519 shared secret and handshake salt. DK1 is the AES-256-GCM key; DK2 regenerates SHAKE-128 positions."
                if mode == "ECDHE"
                else "DK1 is the AES-256-GCM key; DK2 regenerates SHAKE-128 positions."
            ),
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
            "positions": positions,
            "rows": _embedding_rows(mapped, positions),
            "key_source": (
                "ECDHE-derived DK2 (not displayed)"
                if mode == "ECDHE"
                else "DK2 (not displayed)"
            ),
        },
    }
    if on_stage is not None:
        for name, data in stages.items():
            on_stage(name, data)

    stages["embedding"] = {
        "status": "running",
        "topic": topic,
        "model": get_model_name(),
        "temperature": float(temperature),
        "top_k": top_k,
        "mapped_character_count": len(mapped),
        "current_stage": "Loading model",
        "message": "Loading the configured model and initializing the embedder.",
    }
    if on_stage is not None:
        on_stage("embedding", stages["embedding"])

    try:
        from server_a import _safe_initial_story
        from app.llm.embedder import EmbedderLLM
        from app.llm.generator import LLMGenerator

        generator = LLMGenerator()
        embedder = EmbedderLLM(llm_generator=generator)

        def report_embedding_progress(completed: int, total: int, status: str) -> None:
            if status == "embedded":
                current_character = completed
                current_stage = "Character embedded"
                message = f"Embedded character {completed}/{total} successfully."
            elif status == "embedding":
                current_character = completed + 1
                current_stage = "Generating candidate"
                message = (
                    f"Generating candidate for character "
                    f"{current_character}/{total}."
                )
            else:
                current_character = completed + 1
                current_stage = "Retrying candidate generation"
                message = (
                    f"{status.capitalize()} for character "
                    f"{current_character}/{total}; generating another candidate."
                )

            stats = getattr(embedder, "_embedding_stats", {})
            progress = {
                **stages["embedding"],
                "status": "running",
                "current_stage": current_stage,
                "current_character": current_character,
                "completed_characters": completed,
                "total_characters": total,
                "message": message,
                "llm_calls": stats.get("llm_calls"),
                "candidate_evaluations": stats.get("candidate_evaluations"),
                "retries": stats.get("retries"),
            }
            stages["embedding"] = progress
            if on_stage is not None:
                on_stage("embedding", progress)

        def report_embedding_event(event: dict[str, object]) -> None:
            _log_embedding_progress(event)

        weight_progress_hook = _install_transformers_weight_progress_hook(
            report_embedding_event
        )
        try:
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
                progress_callback=report_embedding_progress,
                progress_event_callback=report_embedding_event,
            )
        finally:
            try:
                _finish_weight_progress_line()
            finally:
                try:
                    _release_embedding_terminal_lock()
                finally:
                    _restore_transformers_weight_progress_hook(weight_progress_hook)
    except (ImportError, OSError, RuntimeError, ValueError, MemoryError) as error:
        embedding_logger.exception("Real LLM embedding failed in web sender.")
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
    stages["embedding"] = {
        **stages["embedding"],
        "status": "success",
        "model": str(generator.model_name),
        "target_positions": actual_positions,
        "attempts": result.attempts,
        "llm_calls": stats.get("llm_calls"),
        "candidate_evaluations": stats.get("candidate_evaluations"),
        "retries": stats.get("retries"),
    }
    stages["carrier"] = {
        "status": "success",
        "text": result.story,
        "length": len(result.story),
        "verified": embedded_ok,
        "notice": (
            "Carrier text is intended to conceal an encrypted payload in generated text; "
            "this does not guarantee undetectability."
        ),
    }
    stages["sender_verification"] = {
        "status": "success",
        "extracted": extracted,
        "authentication": "success",
    }
    if on_stage is not None:
        for name in ("embedding", "carrier", "sender_verification"):
            on_stage(name, stages[name])

    if claimed_session_id is not None and ecdhe_session is not None:
        _complete_ecdhe_sender_session(
            claimed_session_id,
            ecdhe_session,
            nonce_hex=encrypted["nonce"].hex(),
            mapped_length=len(mapped),
        )

    return {
        "mode": mode,
        "model": str(generator.model_name),
        "session": {
            "salt_hex": salt.hex(),
            "nonce_hex": encrypted["nonce"].hex(),
            "mapped_length": len(mapped),
            **(
                {"ecdhe_session_id": claimed_session_id}
                if claimed_session_id is not None
                else {}
            ),
        },
        "stages": stages,
    }


def run_receiver(
    *,
    carrier_text: object,
    password: object = None,
    salt_hex: object = None,
    nonce_hex: object = None,
    mapped_length: object = None,
    mode: object = "PSK",
    ecdhe_session_id: object = None,
    on_stage: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Regenerate positions, extract the payload, and authenticate it."""

    carrier_text = _require_text(carrier_text, "Received carrier text")
    if not isinstance(mode, str) or mode not in {"PSK", "ECDHE"}:
        raise DemoPipelineError("input", "Key-establishment mode must be PSK or ECDHE.")
    if mode == "PSK":
        password = _require_text(password, "Password")
        salt_hex = _require_text(salt_hex, "Salt")
        nonce_hex = _require_text(nonce_hex, "Nonce")
    elif password is not None:
        raise DemoPipelineError("input", "ECDHE mode does not accept a password.")
    if (
        not isinstance(mapped_length, int)
        or isinstance(mapped_length, bool)
        or mapped_length <= 0
        or mapped_length % 2
        or mapped_length > MAX_MAPPED_CHARACTERS
    ):
        raise DemoPipelineError("input", "Mapped payload length is invalid.")
    if mode == "PSK":
        try:
            salt = bytes.fromhex(salt_hex)
            nonce = bytes.fromhex(nonce_hex)
            if not salt:
                raise ValueError("empty salt")
            if len(nonce) != 12:
                raise ValueError("invalid nonce length")
        except ValueError as error:
            raise DemoPipelineError("input", "Salt or nonce metadata is invalid.") from error

    _emit_stage(
        on_stage,
        "key_derivation",
        {"status": "running", "message": "Preparing receiver key material."},
    )
    try:
        if mode == "PSK":
            dk1, dk2 = derive_keys(password, salt)
        else:
            salt, dk1, dk2 = _get_ecdhe_receiver_keys(
                ecdhe_session_id,
                salt_hex=salt_hex,
                nonce_hex=nonce_hex,
                mapped_length=mapped_length,
            )
        nonce = bytes.fromhex(nonce_hex) if isinstance(nonce_hex, str) else b""
        if len(nonce) != 12:
            raise ValueError("invalid nonce length")
        associated_data = payload_associated_data(mapped_length)
    except DemoPipelineError as error:
        _emit_stage(
            on_stage,
            "key_derivation",
            {"status": "failed", "message": str(error)},
        )
        raise
    except (TypeError, ValueError, OverflowError) as error:
        _emit_stage(
            on_stage,
            "key_derivation",
            {"status": "failed", "message": "Receiver key state is invalid or incomplete."},
        )
        raise DemoPipelineError(
            "key_derivation", "Key derivation or session validation failed."
        ) from error

    _emit_stage(
        on_stage,
        "key_derivation",
        {
            "status": "success",
            "algorithm": f"PBKDF2-HMAC-{DEFAULT_PBKDF2_CONFIG.hash_name.upper()}",
            "iterations": get_pbkdf2_iterations(),
            "dk1_length": len(dk1),
            "dk2_length": len(dk2),
            "key_source": "ECDHE shared secret" if mode == "ECDHE" else "PSK",
        },
    )
    _emit_stage(
        on_stage,
        "extraction",
        {"status": "running", "message": "Regenerating target positions and extracting the carrier."},
    )
    try:
        positions = positions_for_payload(dk2, mapped_length)
        extracted = Extractor.extract(carrier_text, positions).upper()
        if len(extracted) != mapped_length:
            raise DemoPipelineError(
                "extraction",
                "Carrier text is too short or does not contain every expected target character.",
            )
        _emit_stage(
            on_stage,
            "extraction",
            {"status": "success", "positions": positions, "extracted": extracted},
        )
    except DemoPipelineError as error:
        _emit_stage(
            on_stage,
            "extraction",
            {"status": "failed", "message": str(error)},
        )
        raise
    except (TypeError, ValueError, OverflowError) as error:
        _emit_stage(
            on_stage,
            "extraction",
            {"status": "failed", "message": "Position regeneration or extraction failed."},
        )
        raise DemoPipelineError("extraction", "Position regeneration or extraction failed.") from error

    try:
        _emit_stage(
            on_stage,
            "inverse_mapping",
            {"status": "running", "message": "Reconstructing tag || ciphertext from h4 characters."},
        )
        recovered_enc = character_sequence_to_aead(extracted)
        if len(recovered_enc) < 16:
            raise ValueError("payload is shorter than an authentication tag")
    except (TypeError, ValueError) as error:
        _emit_stage(
            on_stage,
            "inverse_mapping",
            {"status": "failed", "message": "Extracted characters are not a valid h4 AEAD payload."},
        )
        raise DemoPipelineError("inverse_mapping", "Extracted characters are not a valid h4 AEAD payload.") from error

    ciphertext = recovered_enc[16:]
    tag = recovered_enc[:16]
    inverse_mapping_stage = {
        "status": "success",
        "payload_hex": recovered_enc.hex(),
        "tag_hex": tag.hex(),
        "ciphertext_hex": ciphertext.hex(),
        "tag_length": len(tag),
        "ciphertext_length": len(ciphertext),
    }
    _emit_stage(on_stage, "inverse_mapping", inverse_mapping_stage)
    _emit_stage(
        on_stage,
        "authentication",
        {"status": "running", "message": "Authenticating and decrypting with AES-256-GCM."},
    )
    try:
        plaintext = decrypt(
            ciphertext=ciphertext,
            tag=tag,
            nonce=nonce,
            dk1=dk1,
            associated_data=associated_data,
        )
    except AEADError:
        authentication_failure = {
            "status": "failed",
            "message": "AES-256-GCM authentication failed. No plaintext was recovered.",
        }
        _emit_stage(on_stage, "authentication", authentication_failure)
        return {
            "mode": mode,
            "stages": {
                "key_derivation": {
                    "status": "success",
                    "algorithm": f"PBKDF2-HMAC-{DEFAULT_PBKDF2_CONFIG.hash_name.upper()}",
                    "iterations": get_pbkdf2_iterations(),
                    "dk1_length": len(dk1),
                    "dk2_length": len(dk2),
                    "key_source": "ECDHE shared secret" if mode == "ECDHE" else "PSK",
                },
                "extraction": {
                    "status": "success",
                    "positions": positions,
                    "extracted": extracted,
                },
                "inverse_mapping": inverse_mapping_stage,
                "authentication": authentication_failure,
            },
        }

    authentication_success = {
        "status": "success",
        "message": "AES-256-GCM authentication succeeded.",
        "plaintext": plaintext.decode("utf-8"),
    }
    _emit_stage(on_stage, "authentication", authentication_success)
    return {
        "mode": mode,
        "stages": {
            "key_derivation": {
                "status": "success",
                "algorithm": f"PBKDF2-HMAC-{DEFAULT_PBKDF2_CONFIG.hash_name.upper()}",
                "iterations": get_pbkdf2_iterations(),
                "dk1_length": len(dk1),
                "dk2_length": len(dk2),
                "key_source": "ECDHE shared secret" if mode == "ECDHE" else "PSK",
            },
            "extraction": {
                "status": "success",
                "positions": positions,
                "extracted": extracted,
            },
            "inverse_mapping": inverse_mapping_stage,
            "authentication": authentication_success,
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
