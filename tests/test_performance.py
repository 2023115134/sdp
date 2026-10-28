from app.performance import (
    PROFILE_SPECS,
    benchmark_to_csv,
    run_auth_negative_tests,
    run_benchmark,
)
from app.crypto.aead import encrypt


def test_run_benchmark_reports_recovery_and_summary_metrics():
    report = run_benchmark(profile="short", trials=2, mode="PSK", use_mock_llm=True)

    assert report["mode"] == "PSK"
    assert report["summary"]["total_trials"] == 2
    assert report["summary"]["successful_trials"] == 2
    assert report["summary"]["recovery_rate"] == 1.0
    assert report["summary"]["median_end_to_end_ms"] >= 0.0
    assert report["summary"]["mean_embedding_throughput"] >= 0.0
    assert all(row["success"] for row in report["trial_results"])
    assert report["trial_results"][0]["message_size_chars"] == PROFILE_SPECS["short"]["message_size_chars"]


def test_failed_trial_counts_toward_recovery_rate():
    report = run_benchmark(
        profile="short",
        trials=2,
        mode="PSK",
        use_mock_llm=True,
        force_fail_trial_indices={2},
    )

    assert report["summary"]["successful_trials"] == 1
    assert report["summary"]["failed_trials"] == 1
    assert report["summary"]["recovery_rate"] == 0.5
    assert report["trial_results"][1]["success"] is False
    assert report["trial_results"][1]["failure_category"] == "recovery_mismatch"


def test_benchmark_csv_includes_trial_rows_and_auth_rejection_columns():
    report = run_benchmark(profile="short", trials=1, mode="PSK", use_mock_llm=True)
    csv_text = benchmark_to_csv({"trial_results": report["trial_results"]})

    assert csv_text.startswith("profile,message_size_chars")
    assert "authentication_negative_tests" in csv_text
    assert "wrong_key_blocked" in csv_text
    assert "success" in csv_text


def test_auth_negative_checks_block_bad_inputs():
    plaintext = b"benchmark-negative-check"
    dk1 = bytes(range(32))
    encrypted = encrypt(plaintext, dk1, b"benchmark-aad")
    result = run_auth_negative_tests(plaintext=plaintext, dk1=dk1, encrypted=encrypted, mode="PSK")

    assert set(result) == {
        "wrong_key_blocked",
        "tampered_ciphertext_blocked",
        "tampered_tag_blocked",
        "malformed_payload_blocked",
    }
    assert all(result.values())
