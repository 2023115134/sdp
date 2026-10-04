import builtins

from app.demo import _run_ecdhe_crypto_only


def test_crypto_only_ecdhe_pipeline_without_llm(monkeypatch, capsys):
    original_import = builtins.__import__

    def reject_llm_import(name, *args, **kwargs):
        if name.startswith("app.llm") or name.startswith("transformers"):
            raise AssertionError(f"Crypto-only mode imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_llm_import)
    result = _run_ecdhe_crypto_only("play", "hi")
    output = capsys.readouterr().out

    assert result.alice_shared_secret == result.bob_shared_secret
    assert len(result.dk1) == 32
    assert len(result.dk2) == 32
    assert len(result.encrypted["tag"]) == 16
    assert result.encrypted["enc"] == (
        result.encrypted["tag"] + result.encrypted["ciphertext"]
    )
    assert result.mapped_payload
    assert len(result.positions) == len(result.mapped_payload)
    assert all(
        left < right
        for left, right in zip(result.positions, result.positions[1:])
    )
    assert result.extracted_mapped_payload == result.mapped_payload
    assert result.recovered_aead_payload == result.encrypted["enc"]
    assert result.recovered_plaintext == b"hi"
    assert "CRYPTO-ONLY VERIFICATION" in output
    assert "# No LLM inference was executed." in output
    assert "Final result: PASS" in output
