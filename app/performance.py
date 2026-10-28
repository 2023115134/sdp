"""Benchmark execution-time measurements for the CovertComm-LLM pipeline."""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
import time
from collections import OrderedDict
from typing import Any, Iterable

from app.crypto.aead import AEADError, decrypt, encrypt
from app.crypto.aead_mapping import aead_to_character_sequence, character_sequence_to_aead
from app.crypto.ecdhe import derive_ecdhe_keys, derive_shared_secret, generate_key_pair
from app.crypto.key_derivation import derive_keys, generate_salt
from app.crypto.mapping import CharacterMap
from app.crypto.position_generator import generate_positions
from app.extraction.extractor import Extractor
from app.llm.embedder import EmbedderLLM
from app.llm.generator import LLMGenerator

PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "short": {
        "label": "Short",
        "message_size_chars": 32,
        "description": "Small payload benchmark",
        "approx_trials": 3,
    },
    "medium": {
        "label": "Medium",
        "message_size_chars": 128,
        "description": "Moderate payload benchmark",
        "approx_trials": 3,
    },
    "long": {
        "label": "Long",
        "message_size_chars": 512,
        "description": "Long payload benchmark",
        "approx_trials": 2,
    },
}

AUTH_NEGATIVE_CHECKS = (
    "wrong_key_blocked",
    "tampered_ciphertext_blocked",
    "tampered_tag_blocked",
    "malformed_payload_blocked",
)


def _synthetic_message(profile_name: str, trial_number: int, message_size_chars: int) -> bytes:
    """Return a deterministic, non-secret benchmark message of the requested length."""
    profile_prefix = {
        "short": "bench-short",
        "medium": "bench-medium",
        "long": "bench-long",
    }.get(profile_name, "bench")
    words = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega" 
        "research demo carrier embedder extractor secure covert message benchmark"
    ).split()

    payload = []
    while len("".join(payload)) < message_size_chars:
        for word in words:
            payload.append(word)
            if len(" ".join(payload)) >= message_size_chars:
                break
        if len(" ".join(payload)) >= message_size_chars:
            break

    text = " ".join(payload)[:message_size_chars]
    suffix = f"-{trial_number}-{profile_prefix}"
    padded = (text + suffix).encode("utf-8")
    if len(padded) >= message_size_chars:
        return padded[:message_size_chars].rstrip(b" ") + b"x"
    return padded.ljust(message_size_chars, b"x")


def _safe_error_summary(error: Exception | None) -> str:
    if error is None:
        return "unknown failure"
    message = str(error).strip()
    if not message:
        return type(error).__name__
    return message[:200]


def _profile_message_sizes(profile_name: str, trial_number: int) -> tuple[int, int]:
    spec = PROFILE_SPECS[profile_name]
    message_bytes = _synthetic_message(profile_name, trial_number, spec["message_size_chars"])
    return len(message_bytes.decode("utf-8")), len(message_bytes)


def _magnitude_from_ns(value: float) -> float:
    return value * 1000.0 if value == 0 else value


def _median_or_zero(values: Iterable[float]) -> float:
    items = [float(value) for value in values if value is not None]
    if not items:
        return 0.0
    return statistics.median(items)


def _summarize_numeric(values: Iterable[float]) -> dict[str, float]:
    items = [float(value) for value in values if value is not None]
    if not items:
        return {
            "mean": 0.0,
            "median": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
        }
    return {
        "mean": sum(items) / len(items),
        "median": _median_or_zero(items),
        "minimum": min(items),
        "maximum": max(items),
    }


def _mock_embed_story(
    topic: str,
    characters: str,
    positions: list[int],
    initial_story: str = "",
) -> str:
    """Insert characters at exact target positions for deterministic mock embedding."""
    story = (initial_story or topic).strip() or "A quiet benchmark story"
    for character, position in zip(characters, positions):
        if len(story) <= position:
            story = story.ljust(position + 1, " ")
        story = story[:position] + character + story[position + 1 :]
    return story


def run_auth_negative_tests(
    plaintext: bytes | None = None,
    dk1: bytes | None = None,
    encrypted: dict[str, bytes] | None = None,
    *,
    mode: str = "PSK",
) -> dict[str, bool]:
    """Exercise authentication rejection paths without exposing any secret data."""
    if encrypted is None:
        if plaintext is None:
            plaintext = b"benchmark-auth-check"
        if dk1 is None:
            dk1 = bytes(range(32))
        encrypted = encrypt(plaintext, dk1, b"benchmark-aad")

    result: dict[str, bool] = {
        "wrong_key_blocked": False,
        "tampered_ciphertext_blocked": False,
        "tampered_tag_blocked": False,
        "malformed_payload_blocked": False,
    }

    wrong_key = bytes((byte + 1) % 256 for byte in dk1)
    try:
        decrypt(
            ciphertext=encrypted["ciphertext"],
            tag=encrypted["tag"],
            nonce=encrypted["nonce"],
            dk1=wrong_key,
            associated_data=b"benchmark-aad",
        )
    except (AEADError, ValueError, TypeError):
        result["wrong_key_blocked"] = True

    tampered_ciphertext = bytearray(encrypted["ciphertext"])
    if tampered_ciphertext:
        tampered_ciphertext[0] ^= 0xFF
    try:
        decrypt(
            ciphertext=bytes(tampered_ciphertext),
            tag=encrypted["tag"],
            nonce=encrypted["nonce"],
            dk1=dk1,
            associated_data=b"benchmark-aad",
        )
    except (AEADError, ValueError, TypeError):
        result["tampered_ciphertext_blocked"] = True

    tampered_tag = bytearray(encrypted["tag"])
    if tampered_tag:
        tampered_tag[0] ^= 0xFF
    try:
        decrypt(
            ciphertext=encrypted["ciphertext"],
            tag=bytes(tampered_tag),
            nonce=encrypted["nonce"],
            dk1=dk1,
            associated_data=b"benchmark-aad",
        )
    except (AEADError, ValueError, TypeError):
        result["tampered_tag_blocked"] = True

    try:
        _ = character_sequence_to_aead("NOT_VALID")
    except (ValueError, TypeError):
        result["malformed_payload_blocked"] = True

    return result


def _coerce_mode(mode: str) -> str:
    normalized = (mode or "PSK").strip().upper()
    if normalized not in {"PSK", "ECDHE"}:
        raise ValueError("mode must be 'PSK' or 'ECDHE'")
    return normalized


def benchmark_to_csv(report: dict[str, Any]) -> str:
    """Export benchmark trial rows as a CSV string without secret material."""
    rows = report.get("trial_results", [])
    fieldnames = [
        "profile",
        "message_size_chars",
        "message_size_bytes",
        "mode",
        "trial_number",
        "llm_execution_type",
        "model_loading_included",
        "key_derivation_ms",
        "encryption_ms",
        "payload_transformation_ms",
        "position_generation_ms",
        "llm_embedding_ms",
        "extraction_ms",
        "decryption_ms",
        "total_duration_ms",
        "carrier_length_chars",
        "carrier_word_count",
        "embedding_throughput_chars_per_second",
        "success",
        "failure_category",
        "error_summary",
        "authentication_negative_tests",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        auth_negative_tests = row.get("authentication_negative_tests") or row.get("auth_negative_tests") or {}
        writer.writerow(
            {
                "profile": row.get("profile"),
                "message_size_chars": row.get("message_size_chars"),
                "message_size_bytes": row.get("message_size_bytes"),
                "mode": row.get("mode"),
                "trial_number": row.get("trial_number"),
                "llm_execution_type": row.get("llm_execution_type"),
                "model_loading_included": row.get("model_loading_included"),
                "key_derivation_ms": row.get("key_derivation_ms"),
                "encryption_ms": row.get("encryption_ms"),
                "payload_transformation_ms": row.get("payload_transformation_ms"),
                "position_generation_ms": row.get("position_generation_ms"),
                "llm_embedding_ms": row.get("llm_embedding_ms"),
                "extraction_ms": row.get("extraction_ms"),
                "decryption_ms": row.get("decryption_ms"),
                "total_duration_ms": row.get("total_duration_ms"),
                "carrier_length_chars": row.get("carrier_length_chars"),
                "carrier_word_count": row.get("carrier_word_count"),
                "embedding_throughput_chars_per_second": row.get("embedding_throughput_chars_per_second"),
                "success": row.get("success"),
                "failure_category": row.get("failure_category"),
                "error_summary": row.get("error_summary"),
                "authentication_negative_tests": json.dumps(auth_negative_tests, sort_keys=True),
            }
        )
    return output.getvalue()


def _generate_key_material(mode: str, profile_name: str, trial_number: int) -> tuple[bytes, bytes, bytes, bytes, bytes, dict[str, Any]]:
    if mode == "PSK":
        password = f"benchmark-{profile_name}-{trial_number}"
        salt = generate_salt(16)
        start = time.perf_counter()
        dk1, dk2 = derive_keys(password, salt)
        key_duration = time.perf_counter() - start
        return dk1, dk2, salt, password.encode("utf-8"), b"", {"key_derivation_ms": key_duration * 1000.0, "salt_hex": salt.hex()}

    alice_private, alice_public = generate_key_pair()
    bob_private, bob_public = generate_key_pair()
    alice_shared = derive_shared_secret(alice_private, bob_public)
    bob_shared = derive_shared_secret(bob_private, alice_public)
    if alice_shared != bob_shared:
        raise RuntimeError("ECDHE shared secret mismatch")
    salt = generate_salt(16)
    start = time.perf_counter()
    dk1, dk2 = derive_ecdhe_keys(alice_shared, salt)
    key_duration = time.perf_counter() - start
    return dk1, dk2, salt, alice_public, bob_public, {"key_derivation_ms": key_duration * 1000.0, "salt_hex": salt.hex()}


def _run_trial(
    *,
    profile_name: str,
    trial_number: int,
    mode: str,
    use_mock_llm: bool | None = None,
    force_fail: bool = False,
) -> dict[str, Any]:
    mode = _coerce_mode(mode)
    spec = PROFILE_SPECS[profile_name]
    message_size_chars = spec["message_size_chars"]
    assert message_size_chars > 0
    plaintext = _synthetic_message(profile_name, trial_number, message_size_chars)

    total_started = time.perf_counter()
    try:
        dk1, dk2, salt, _, _, key_meta = _generate_key_material(mode, profile_name, trial_number)
        start_key = time.perf_counter()
        key_derivation_ms = (time.perf_counter() - total_started) * 1000.0
        _ = key_derivation_ms
        associated_data = b"llm-shield/performance/v1"
        encryption_start = time.perf_counter()
        encrypted = encrypt(plaintext, dk1, associated_data)
        encryption_ms = (time.perf_counter() - encryption_start) * 1000.0

        payload_start = time.perf_counter()
        mapped = aead_to_character_sequence(encrypted["enc"])
        payload_ms = (time.perf_counter() - payload_start) * 1000.0

        position_start = time.perf_counter()
        positions = generate_positions(
            key_material=dk2,
            number_of_positions=len(mapped),
            offset_do=32,
            max_story_length=max(4000, 32 + len(mapped) * 64),
            min_gap=1,
        )
        position_ms = (time.perf_counter() - position_start) * 1000.0

        topic = f"{profile_name} benchmark trail with secure carrier generation"
        initial_story = (topic[: max(1, positions[0] - 1)] if positions else topic).rstrip() or "A quiet benchmark story"
        while positions and len(initial_story) >= positions[0]:
            initial_story = initial_story[:-1].rstrip() or "A quiet benchmark story"

        model_loading_included = False
        llm_type = "mock"
        if use_mock_llm is None:
            use_mock_llm = False
        embedding_start = time.perf_counter()
        if use_mock_llm:
            story = _mock_embed_story(topic, mapped, positions, initial_story)
            llm_type = "mock"
        else:
            try:
                model_loading_included = True
                embedder = EmbedderLLM(llm_generator=LLMGenerator(), character_map=CharacterMap())
                result = embedder.embed(
                    topic=topic,
                    characters=mapped,
                    positions=positions,
                    initial_story=initial_story,
                    temperature=0.7,
                    top_k=20,
                    max_new_tokens=32,
                    max_attempts=150,
                    max_retries=2,
                    verbose=False,
                )
                story = result.story
                llm_type = "real"
            except Exception:
                story = _mock_embed_story(topic, mapped, positions, initial_story)
                llm_type = "mock-fallback"
        embedding_ms = (time.perf_counter() - embedding_start) * 1000.0

        extraction_start = time.perf_counter()
        extracted = Extractor.extract(story, positions)
        extraction_ms = (time.perf_counter() - extraction_start) * 1000.0

        decrypt_start = time.perf_counter()
        recovered_plaintext = decrypt(
            ciphertext=character_sequence_to_aead(extracted)[16:],
            tag=character_sequence_to_aead(extracted)[:16],
            nonce=encrypted["nonce"],
            dk1=dk1,
            associated_data=associated_data,
        )
        decrypt_ms = (time.perf_counter() - decrypt_start) * 1000.0

        if force_fail:
            recovered_plaintext = b"forced-failure"

        success = recovered_plaintext == plaintext
        failure_category = None
        error_summary = None
        if not success:
            failure_category = "recovery_mismatch"
            error_summary = "Recovered plaintext did not match the synthetic benchmark message."
        auth_negative_tests = run_auth_negative_tests(
            plaintext=plaintext,
            dk1=dk1,
            encrypted=encrypted,
            mode=mode,
        )
        total_duration_ms = (time.perf_counter() - total_started) * 1000.0

        return {
            "profile": profile_name,
            "mode": mode,
            "trial_number": trial_number,
            "message_size_chars": message_size_chars,
            "message_size_bytes": len(plaintext),
            "key_derivation_ms": key_meta.get("key_derivation_ms", 0.0),
            "encryption_ms": encryption_ms,
            "payload_transformation_ms": payload_ms,
            "position_generation_ms": position_ms,
            "llm_embedding_ms": embedding_ms,
            "extraction_ms": extraction_ms,
            "decryption_ms": decrypt_ms,
            "total_duration_ms": total_duration_ms,
            "carrier_length_chars": len(story),
            "carrier_word_count": len(story.split()),
            "embedding_throughput_chars_per_second": (
                (len(mapped) / max(embedding_ms / 1000.0, 1e-9)) if embedding_ms > 0 else 0.0
            ),
            "llm_execution_type": llm_type,
            "model_loading_included": model_loading_included,
            "auth_negative_tests": auth_negative_tests,
            "success": success,
            "failure_category": failure_category,
            "error_summary": error_summary,
            "carrier_text_length": len(story),
            "seed": f"{profile_name}-{trial_number}",
        }
    except Exception as error:
        total_duration_ms = (time.perf_counter() - total_started) * 1000.0
        return {
            "profile": profile_name,
            "mode": mode,
            "trial_number": trial_number,
            "message_size_chars": spec["message_size_chars"],
            "message_size_bytes": len(_synthetic_message(profile_name, trial_number, spec["message_size_chars"])),
            "key_derivation_ms": 0.0,
            "encryption_ms": 0.0,
            "payload_transformation_ms": 0.0,
            "position_generation_ms": 0.0,
            "llm_embedding_ms": 0.0,
            "extraction_ms": 0.0,
            "decryption_ms": 0.0,
            "total_duration_ms": total_duration_ms,
            "carrier_length_chars": 0,
            "carrier_word_count": 0,
            "embedding_throughput_chars_per_second": 0.0,
            "llm_execution_type": "mock",
            "model_loading_included": False,
            "auth_negative_tests": {name: False for name in AUTH_NEGATIVE_CHECKS},
            "success": False,
            "failure_category": "runtime_error",
            "error_summary": _safe_error_summary(error),
            "carrier_text_length": 0,
            "seed": f"{profile_name}-{trial_number}",
        }


def _profile_summary(profile_trials: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(profile_trials)
    successful = sum(1 for row in profile_trials if row.get("success") is True)
    summary = {
        "profile": profile_trials[0]["profile"],
        "mode": profile_trials[0]["mode"],
        "total_trials": total,
        "successful_trials": successful,
        "failed_trials": total - successful,
        "recovery_rate": (successful / total) if total else 0.0,
        "median_end_to_end_ms": _median_or_zero(row["total_duration_ms"] for row in profile_trials),
        "mean_end_to_end_ms": sum(row["total_duration_ms"] for row in profile_trials) / total if total else 0.0,
        "min_end_to_end_ms": min(row["total_duration_ms"] for row in profile_trials) if total else 0.0,
        "max_end_to_end_ms": max(row["total_duration_ms"] for row in profile_trials) if total else 0.0,
        "median_embedding_throughput": _median_or_zero(row["embedding_throughput_chars_per_second"] for row in profile_trials),
        "mean_embedding_throughput": sum(row["embedding_throughput_chars_per_second"] for row in profile_trials) / total if total else 0.0,
        "average_carrier_length_chars": sum(row["carrier_length_chars"] for row in profile_trials) / total if total else 0.0,
        "average_carrier_word_count": sum(row["carrier_word_count"] for row in profile_trials) / total if total else 0.0,
        "stage_durations": {
            key: _summarize_numeric(row[key] for row in profile_trials)
            for key in (
                "key_derivation_ms",
                "encryption_ms",
                "payload_transformation_ms",
                "position_generation_ms",
                "llm_embedding_ms",
                "extraction_ms",
                "decryption_ms",
            )
        },
        "auth_negative_tests": {
            key: sum(1 for row in profile_trials if row.get("auth_negative_tests", {}).get(key, False))
            for key in AUTH_NEGATIVE_CHECKS
        },
    }
    return summary


def run_benchmark(
    *,
    profile: str = "short",
    trials: int = 3,
    mode: str = "PSK",
    use_mock_llm: bool | None = None,
    force_fail_trial_indices: set[int] | None = None,
    include_auth_negative_tests: bool = True,
) -> dict[str, Any]:
    """Run benchmark trials for a single profile or all profiles, recording failures transparently."""
    force_fail_trial_indices = force_fail_trial_indices or set()
    normalized_mode = _coerce_mode(mode)
    if not isinstance(trials, int) or trials <= 0:
        raise ValueError("trials must be a positive integer")

    selected_profiles = [profile] if profile != "all" else list(PROFILE_SPECS)
    results: list[dict[str, Any]] = []
    profile_summaries: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for profile_name in selected_profiles:
        if profile_name not in PROFILE_SPECS:
            raise ValueError(f"Unsupported benchmark profile: {profile_name}")
        for trial_number in range(1, trials + 1):
            row = _run_trial(
                profile_name=profile_name,
                trial_number=trial_number,
                mode=normalized_mode,
                use_mock_llm=use_mock_llm,
                force_fail=trial_number in force_fail_trial_indices,
            )
            if include_auth_negative_tests:
                row["auth_negative_tests"] = row.get("auth_negative_tests", {name: False for name in AUTH_NEGATIVE_CHECKS})
                row["authentication_negative_tests"] = row["auth_negative_tests"]
            results.append(row)
        profile_summaries[profile_name] = _profile_summary(
            [row for row in results if row["profile"] == profile_name]
        )

    summary_all = {
        "total_trials": len(results),
        "successful_trials": sum(1 for row in results if row.get("success") is True),
        "failed_trials": sum(1 for row in results if row.get("success") is not True),
        "recovery_rate": (
            sum(1 for row in results if row.get("success") is True) / len(results)
            if results
            else 0.0
        ),
        "median_end_to_end_ms": _median_or_zero(row["total_duration_ms"] for row in results),
        "mean_end_to_end_ms": sum(row["total_duration_ms"] for row in results) / len(results) if results else 0.0,
        "min_end_to_end_ms": min((row["total_duration_ms"] for row in results), default=0.0),
        "max_end_to_end_ms": max((row["total_duration_ms"] for row in results), default=0.0),
        "mean_embedding_throughput": sum(row["embedding_throughput_chars_per_second"] for row in results) / len(results) if results else 0.0,
        "average_carrier_length_chars": sum(row["carrier_length_chars"] for row in results) / len(results) if results else 0.0,
    }

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": normalized_mode,
        "profile": profile,
        "summary": summary_all,
        "profile_results": profile_summaries,
        "trial_results": results,
        "csv": benchmark_to_csv({"trial_results": results}),
        "metadata": {
            "profiles": PROFILE_SPECS,
            "selected_profiles": selected_profiles,
            "trials_per_profile": trials,
            "llm_execution_note": "Real LLM results are recorded when available; if the model is unavailable or fails, the benchmark falls back to deterministic mock embedding and records the fallback explicitly.",
            "mode_operational": normalized_mode,
        },
    }
    return report


__all__ = [
    "PROFILE_SPECS",
    "AUTH_NEGATIVE_CHECKS",
    "benchmark_to_csv",
    "run_auth_negative_tests",
    "run_benchmark",
]
