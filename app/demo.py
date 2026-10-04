"""Main research prototype demo for PSK and ECDHE modes.

Both modes use the same covert pipeline:

    key establishment
      -> key derivation / session-key agreement
      -> AES-256-GCM encryption
      -> h4 mapping
      -> SHAKE-128 target positions
      -> EmbedderLLM carrier generation
      -> extraction
      -> inverse mapping
      -> AES-GCM verification/decryption
      -> original message
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Sequence

from app.crypto.aead import decrypt, encrypt
from app.crypto.aead_mapping import (
    aead_to_character_sequence,
    character_sequence_to_aead,
)
from app.crypto.ecdhe import (
    derive_ecdhe_keys,
    derive_shared_secret,
    generate_key_pair,
)
from app.crypto.key_derivation import derive_keys, generate_salt
from app.crypto.mapping import CharacterMap
from app.crypto.position_generator import generate_positions


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)


def _derive_position_key(base_key: bytes, label: str) -> bytes:
    """Derive a stable position key from the session or PSK material."""
    return hashlib.sha256(base_key + label.encode("utf-8")).digest()


def _build_safe_initial_story(topic: str, positions: Sequence[int]) -> str:
    """Return a story prefix that is guaranteed to start before the first target."""
    if not positions:
        raise ValueError("positions must not be empty")

    first_position = min(int(position) for position in positions)
    if first_position <= 0:
        raise ValueError("first target position must be positive")

    base = (topic or "A quiet story").strip()
    if not base:
        base = "A quiet story"

    candidate = base[: max(1, first_position - 1)].rstrip()
    if not candidate:
        candidate = "A quiet story"

    while len(candidate) >= first_position:
        candidate = candidate[: max(1, len(candidate) - 1)].rstrip()
        if not candidate:
            candidate = "A quiet story"

    return candidate


def _print_mode_banner(mode_name: str) -> None:
    print("=" * 78)
    print(f"LLM-SHIELD MAIN DEMO: {mode_name}")
    print("=" * 78)


def _run_common_cover_pipeline(
    *,
    topic: str,
    secret: str,
    encryption_key: bytes,
    position_key: bytes,
    initial_story: str = "",
    key_label: str,
    encrypted_packet: dict[str, bytes] | None = None,
    target_positions: list[int] | None = None,
    associated_data: bytes | None = None,
) -> dict[str, object]:
    """Use the existing AES-GCM + mapping + SHAKE-128 + EmbedderLLM pipeline."""

    from app.extraction.extractor import Extractor
    from app.llm.embedder import EmbedderLLM
    from app.llm.generator import LLMGenerator

    class DemoEmbedder(EmbedderLLM):
        """Add demo-only timing around the unchanged embedder operation."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._demo_embedding_started = time.perf_counter()
            self._demo_total = 0

        def _embed_one_character(self, *args, **kwargs):
            character = kwargs.get("character", args[1] if len(args) > 1 else "")
            position = kwargs.get("position", args[2] if len(args) > 2 else 0)
            max_retries = kwargs.get("max_retries", args[6] if len(args) > 6 else 0)
            elapsed = time.perf_counter() - self._demo_embedding_started
            if key_label != "ECDHE":
                print(
                    f"[embedding] character={character!r} position={position} "
                    f"retries<= {max_retries} elapsed={elapsed:.1f}s"
                )
            result = super()._embed_one_character(*args, **kwargs)
            self._demo_completed = getattr(self, "_demo_completed", 0) + 1
            total = getattr(self, "_demo_total", "?")
            elapsed = time.perf_counter() - self._demo_embedding_started
            if key_label != "ECDHE":
                print(
                    f"[embedding] progress={self._demo_completed}/{total} "
                    f"elapsed={elapsed:.1f}s"
                )
            return result

    plaintext = secret.encode("utf-8")
    encrypted = encrypted_packet or encrypt(
        plaintext,
        encryption_key,
        associated_data,
    )
    enc = encrypted["enc"]
    nonce = encrypted["nonce"]
    mapped = aead_to_character_sequence(enc)

    if target_positions is None:
        offset_do = 32
        bit_chunk_size = 5
        max_step = offset_do + ((1 << bit_chunk_size) - 1)
        required_story_length = max(
            5000,
            offset_do + len(mapped) * max_step + 512,
        )

        positions = generate_positions(
            key_material=position_key,
            number_of_positions=len(mapped),
            offset_do=offset_do,
            max_story_length=required_story_length,
            min_gap=1,
        )
    else:
        positions = target_positions
        if len(positions) != len(mapped):
            raise ValueError("target_positions must match mapped payload length")

    safe_initial_story = _build_safe_initial_story(
        initial_story.strip() if initial_story and initial_story.strip() else topic,
        positions,
    )

    generator = LLMGenerator()
    character_map = CharacterMap()
    embedder = DemoEmbedder(llm_generator=generator, character_map=character_map)
    embedder._demo_total = len(mapped)
    if key_label == "ECDHE":
        print("\nLLM Model:")
        print(generator.model_name)
        print("\nEmbedding Payload Length:")
        print(len(mapped))
        print("\nStarting actual EmbedderLLM/Qwen embedding...")
    start = time.perf_counter()
    result = embedder.embed(
        topic=topic,
        characters=mapped,
        positions=positions,
        initial_story=safe_initial_story,
        temperature=0.7,
        top_k=20,
        max_new_tokens=32,
        max_attempts=15000,
        max_retries=3,
    )
    elapsed = time.perf_counter() - start

    extractor = Extractor(position_generator=None, character_map=character_map)
    extracted = extractor.extract(
        cover_text=result.story,
        positions=positions,
    )
    canonical_extracted = extracted.upper()
    recovered_enc = character_sequence_to_aead(canonical_extracted)
    tag = recovered_enc[:16]
    ciphertext = recovered_enc[16:]
    recovered_plaintext = decrypt(
        ciphertext=ciphertext,
        tag=tag,
        nonce=nonce,
        dk1=encryption_key,
        associated_data=associated_data,
    )

    embedding_verified = EmbedderLLM.verify_embedding(
        result.story,
        mapped,
        positions,
    )
    payload_state = (
        recovered_plaintext == plaintext
        and canonical_extracted == mapped
        and recovered_enc == enc
    )
    naturalness_state = embedder._validate_cover_naturalness(
        story=result.story,
        topic=topic,
    )
    cover_state = all(naturalness_state.values())
    overall_state = payload_state and cover_state and embedding_verified

    if key_label == "ECDHE":
        print("\nGenerated Cover Text:")
        print(result.story)
        print("\nEmbedding Positions:")
        print(positions)
        print("\nEmbedded Characters:")
        print(result.embedded_characters)
        print("\nEmbedding Verification:")
        print("PASS" if embedding_verified else "FAIL")
        print("\nExtracting payload from carrier text...")
        print("\nExtracted Characters:")
        print(canonical_extracted)
        print("\nExtracted Payload Length:")
        print(len(canonical_extracted))
        print("\nExtraction:")
        print("PASS" if canonical_extracted == mapped else "FAIL")
        print("\nMapped Payload:")
        print(mapped)
        print("\nRecovered AEAD Payload:")
        print(recovered_enc.hex())
        print("\nInverse Mapping:")
        print("PASS" if recovered_enc == enc else "FAIL")
        print("\nPayload Recovery:")
        print("PASS" if payload_state else "FAIL")
        print("\nAuthentication Tag Verification:")
        print("PASS" if payload_state else "FAIL")
        print("\nAES-256-GCM Decryption:")
        print("PASS" if recovered_plaintext == plaintext else "FAIL")
        print("\nRecovered Plaintext:")
        print(recovered_plaintext.decode("utf-8"))
        print("\nOriginal Message:")
        print(secret)
        print("\nOriginal == Recovered:")
        print("PASS" if recovered_plaintext == plaintext else "FAIL")
    else:
        print(f"\n[{key_label}] Secure carrier pipeline")
        print(f"[{key_label}] Plaintext:", repr(secret))
        print(f"[{key_label}] Encrypted payload length:", len(enc))
        print(f"[{key_label}] Carrier length:", len(result.story))
        print(f"[{key_label}] Embedding positions:", len(positions))
        print(f"[{key_label}] Payload integrity:", "PASS" if payload_state else "FAIL")
        print(f"[{key_label}] Cover naturalness:", "PASS" if cover_state else "FAIL")
        print(f"[{key_label}] Embedding runtime:", round(elapsed, 2), "seconds")
        print(f"[{key_label}] Recovered plaintext:", recovered_plaintext.decode("utf-8"))
        print(f"[{key_label}] Final result:", "PASS" if overall_state else "FAIL")

    if not overall_state:
        raise RuntimeError(f"[{key_label}] secure covert pipeline failed")

    return {
        "story": result.story,
        "positions": positions,
        "encrypted": encrypted,
        "mapped": mapped,
        "extracted": canonical_extracted,
        "recovered_enc": recovered_enc,
        "payload_state": payload_state,
        "cover_state": cover_state,
        "embedding_verified": embedding_verified,
        "overall_state": overall_state,
        "recovered_plaintext": recovered_plaintext,
        "plaintext": plaintext,
        "key_label": key_label,
    }


def _run_psk_mode(topic: str, secret: str, password: str) -> None:
    _print_mode_banner("PASSWORD / PSK")
    salt = generate_salt()
    dk1, dk2 = derive_keys(password, salt)
    position_key = dk2
    print("[PSK] salt:", salt.hex())
    print("[PSK] dk1 length:", len(dk1), "bytes")
    print("[PSK] dk2 length:", len(dk2), "bytes")
    _run_common_cover_pipeline(
        topic=topic,
        secret=secret,
        encryption_key=dk1,
        position_key=position_key,
        key_label="PSK",
    )


@dataclass(frozen=True)
class ECDHECryptoResult:
    alice_public_key: bytes
    bob_public_key: bytes
    alice_shared_secret: bytes
    bob_shared_secret: bytes
    salt: bytes
    dk1: bytes
    dk2: bytes
    associated_data: bytes
    encrypted: dict[str, bytes]
    mapped_payload: str
    positions: list[int]
    extracted_mapped_payload: str
    recovered_aead_payload: bytes
    recovered_plaintext: bytes


def _execute_ecdhe_crypto_pipeline(secret: str) -> ECDHECryptoResult:
    """Run ECDHE, PBKDF2, AEAD, mapping, positions, and simulated extraction."""

    alice_private, alice_public = generate_key_pair()
    bob_private, bob_public = generate_key_pair()
    alice_shared_secret = derive_shared_secret(alice_private, bob_public)
    bob_shared_secret = derive_shared_secret(bob_private, alice_public)

    if alice_shared_secret != bob_shared_secret:
        raise RuntimeError("ECDHE raw shared secrets do not match")

    salt = generate_salt()
    dk1, dk2 = derive_ecdhe_keys(alice_shared_secret, salt)
    associated_data = b"llm-shield/ecdhe/crypto-only"
    plaintext = secret.encode("utf-8")
    encrypted = encrypt(plaintext, dk1, associated_data)
    mapped_payload = aead_to_character_sequence(encrypted["enc"])

    positions = generate_positions(
        key_material=dk2,
        number_of_positions=len(mapped_payload),
        offset_do=32,
        max_story_length=max(100_000, 32 + len(mapped_payload) * 64),
        min_gap=1,
    )
    position_payload = dict(zip(positions, mapped_payload))
    extracted_mapped_payload = "".join(
        position_payload[position] for position in positions
    )
    recovered_aead_payload = character_sequence_to_aead(
        extracted_mapped_payload
    )

    tag = recovered_aead_payload[:16]
    ciphertext = recovered_aead_payload[16:]
    recovered_plaintext = decrypt(
        ciphertext=ciphertext,
        tag=tag,
        nonce=encrypted["nonce"],
        dk1=dk1,
        associated_data=associated_data,
    )

    return ECDHECryptoResult(
        alice_public_key=alice_public,
        bob_public_key=bob_public,
        alice_shared_secret=alice_shared_secret,
        bob_shared_secret=bob_shared_secret,
        salt=salt,
        dk1=dk1,
        dk2=dk2,
        associated_data=associated_data,
        encrypted=encrypted,
        mapped_payload=mapped_payload,
        positions=positions,
        extracted_mapped_payload=extracted_mapped_payload,
        recovered_aead_payload=recovered_aead_payload,
        recovered_plaintext=recovered_plaintext,
    )


def _run_ecdhe_crypto_only(topic: str, secret: str) -> ECDHECryptoResult:
    """Print a detailed crypto-only verification without loading the LLM stack."""

    started = time.perf_counter()
    result = _execute_ecdhe_crypto_pipeline(secret)
    payload_recovered = (
        result.extracted_mapped_payload == result.mapped_payload
        and result.recovered_aead_payload == result.encrypted["enc"]
    )
    secrets_match = result.alice_shared_secret == result.bob_shared_secret
    keys_valid = len(result.dk1) == len(result.dk2) == 32
    positions_valid = (
        len(result.positions) == len(result.mapped_payload)
        and all(a < b for a, b in zip(result.positions, result.positions[1:]))
    )
    encryption_valid = (
        len(result.encrypted["tag"]) == 16
        and result.encrypted["enc"]
        == result.encrypted["tag"] + result.encrypted["ciphertext"]
    )
    all_valid = (
        secrets_match
        and keys_valid
        and positions_valid
        and encryption_valid
        and payload_recovered
        and result.recovered_plaintext == secret.encode("utf-8")
    )

    print("=" * 78)
    print("CRYPTO-ONLY VERIFICATION")
    print("=" * 78)
    print("[1] INPUT")
    print("Topic:", topic)
    print("Secret message:", secret)

    print("\n[2] ECDHE KEY EXCHANGE")
    print("\nAlice public key:")
    print(result.alice_public_key.hex())
    print("\nBob public key:")
    print(result.bob_public_key.hex())
    print("\nAlice raw shared secret:")
    print(result.alice_shared_secret.hex())
    print("\nBob raw shared secret:")
    print(result.bob_shared_secret.hex())
    print("\nShared secret match:", "PASS" if secrets_match else "FAIL")

    print("\n[3] PBKDF2 KEY DERIVATION")
    print("\nPBKDF2 salt:")
    print(result.salt.hex())
    print("\nPBKDF2 output length:")
    print(f"{len(result.dk1) + len(result.dk2)} bytes")
    print("\nDK1:")
    print(result.dk1.hex())
    print("\nDK1 length:")
    print(f"{len(result.dk1)} bytes")
    print("\nDK2:")
    print(result.dk2.hex())
    print("\nDK2 length:")
    print(f"{len(result.dk2)} bytes")
    print("\nDK1/DK2 derivation:", "PASS" if keys_valid else "FAIL")

    print("\n[4] AES-256-GCM / AEAD")
    print("\nPlaintext:")
    print(secret)
    print("\nAAD:")
    print(result.associated_data.hex())
    print("\nNonce:")
    print(result.encrypted["nonce"].hex())
    print("\nCiphertext:")
    print(result.encrypted["ciphertext"].hex())
    print("\nAuthentication tag:")
    print(result.encrypted["tag"].hex())
    print("\nCiphertext length:")
    print(f"{len(result.encrypted['ciphertext'])} bytes")
    print("\nTag length:")
    print(f"{len(result.encrypted['tag'])} bytes")
    print("\nAEAD encryption:", "PASS" if encryption_valid else "FAIL")

    print("\n[5] PAYLOAD MAPPING")
    print("\nAEAD payload:")
    print(result.encrypted["enc"].hex())
    print("\nMapped payload:")
    print(result.mapped_payload)
    print("\nMapped payload length:")
    print(len(result.mapped_payload))
    print("\nMapping:", "PASS" if payload_recovered else "FAIL")

    print("\n[6] SHAKE-128 POSITION GENERATION")
    print("\nPosition key:")
    print("DK2")
    print("\nNumber of positions:")
    print(len(result.positions))
    print("\nPosition sequence:")
    print(result.positions)
    print("\nPositions strictly increasing:")
    print("PASS" if positions_valid else "FAIL")
    print("\nPosition generation:", "PASS" if positions_valid else "FAIL")

    print("\n[7] CRYPTO-ONLY EXTRACTION")
    print("\nExtracted mapped payload:")
    print(result.extracted_mapped_payload)
    print("\nInverse-mapped AEAD payload:")
    print(result.recovered_aead_payload.hex())
    print("\nPayload recovery:", "PASS" if payload_recovered else "FAIL")

    print("\n[8] AEAD VERIFICATION / DECRYPTION")
    print("\nAuthentication tag verification:")
    print("PASS" if result.recovered_plaintext == secret.encode("utf-8") else "FAIL")
    print("\nRecovered plaintext:")
    print(result.recovered_plaintext.decode("utf-8"))
    print("\nOriginal == recovered:")
    print("PASS" if result.recovered_plaintext == secret.encode("utf-8") else "FAIL")
    print("\n# No LLM inference was executed.")

    elapsed = time.perf_counter() - started
    print(f"\nCrypto-only runtime: {elapsed:.3f} seconds")
    print("Final result:", "PASS" if all_valid else "FAIL")
    if not all_valid:
        raise RuntimeError("ECDHE crypto-only verification failed")
    return result


def _run_ecdhe_mode(topic: str, secret: str) -> None:
    started = time.perf_counter()
    alice_private, alice_public = generate_key_pair()
    bob_private, bob_public = generate_key_pair()
    print("=" * 78)
    print("ECDHE END-TO-END PRESENTATION DEMO")
    print("=" * 78)
    print("\n[1] INPUT")
    print(f"Topic          : {topic}")
    print(f"Secret Message : {secret}")

    print("\n[2] ECDHE KEY EXCHANGE")
    print("\nAlice Public Key:")
    print(alice_public.hex())
    print("\nBob Public Key:")
    print(bob_public.hex())
    alice_shared_secret = derive_shared_secret(alice_private, bob_public)
    bob_shared_secret = derive_shared_secret(bob_private, alice_public)
    shared_secret_matches = alice_shared_secret == bob_shared_secret
    print("\nAlice Raw Shared Secret:")
    print(alice_shared_secret.hex())
    print("\nBob Raw Shared Secret:")
    print(bob_shared_secret.hex())
    print("\nShared Secret Match:")
    print("PASS" if shared_secret_matches else "FAIL")
    if not shared_secret_matches:
        raise RuntimeError("ECDHE raw shared secret mismatch")

    print("\n[3] PBKDF2 KEY DERIVATION")
    salt = generate_salt()
    dk1, dk2 = derive_ecdhe_keys(alice_shared_secret, salt)
    pbkdf2_valid = len(dk1) == 32 and len(dk2) == 32
    print("\nPBKDF2 Salt:")
    print(salt.hex())
    print("\nPBKDF2 Output:")
    print(f"{(len(dk1) + len(dk2)) * 8} bits / {len(dk1) + len(dk2)} bytes")
    print("\nDK1:")
    print(dk1.hex())
    print("\nDK1 Length:")
    print(f"{len(dk1)} bytes")
    print("\nDK2:")
    print(dk2.hex())
    print("\nDK2 Length:")
    print(f"{len(dk2)} bytes")
    print("\nDK1/DK2 Derivation:")
    print("PASS" if pbkdf2_valid else "FAIL")
    if not pbkdf2_valid:
        raise RuntimeError("ECDHE PBKDF2 key derivation produced invalid key lengths")

    print("\n[4] AES-256-GCM / AEAD")
    plaintext = secret.encode("utf-8")
    associated_data = b"llm-shield/carrier-session"
    encrypted = encrypt(plaintext, dk1, associated_data)
    aead_valid = (
        len(encrypted["nonce"]) == 12
        and len(encrypted["tag"]) == 16
        and encrypted["enc"] == encrypted["tag"] + encrypted["ciphertext"]
    )
    print("\nPlaintext:")
    print(secret)
    print("\nAAD:")
    print(associated_data.hex())
    print("\nNonce:")
    print(encrypted["nonce"].hex())
    print("\nCiphertext:")
    print(encrypted["ciphertext"].hex())
    print("\nAuthentication Tag:")
    print(encrypted["tag"].hex())
    print("\nCiphertext Length:")
    print(f"{len(encrypted['ciphertext'])} bytes")
    print("\nAuthentication Tag Length:")
    print(f"{len(encrypted['tag'])} bytes")
    print("\nAEAD Encryption:")
    print("PASS" if aead_valid else "FAIL")
    if not aead_valid:
        raise RuntimeError("ECDHE AES-GCM encryption output is invalid")

    print("\n[5] PAYLOAD MAPPING")
    print("\nOriginal AEAD Payload:")
    print(encrypted["enc"].hex())
    mapped = aead_to_character_sequence(encrypted["enc"])
    mapping_valid = character_sequence_to_aead(mapped) == encrypted["enc"]
    print("\nMapped Payload:")
    print(mapped)
    print("\nMapped Payload Length:")
    print(len(mapped))
    print("\nMapping:")
    print("PASS" if mapping_valid else "FAIL")
    if not mapping_valid:
        raise RuntimeError("ECDHE AEAD payload mapping did not round-trip")

    print("\n[6] SHAKE-128 POSITION GENERATION")
    print("\nPosition Key:")
    print("DK2")
    print("\nSHAKE-128:")
    print("ACTIVE")
    positions = generate_positions(
        key_material=dk2,
        number_of_positions=len(mapped),
        offset_do=32,
        max_story_length=100_000,
        min_gap=1,
    )
    positions_valid = (
        len(positions) == len(mapped)
        and all(left < right for left, right in zip(positions, positions[1:]))
    )
    print("\nNumber of Target Positions:")
    print(len(positions))
    print("\nTarget Position Sequence:")
    print(positions)
    print("\nPositions Strictly Increasing:")
    print("PASS" if positions_valid else "FAIL")
    print("\nPosition Generation:")
    print("PASS" if positions_valid else "FAIL")
    if not positions_valid:
        raise RuntimeError("ECDHE target positions are invalid")

    print("\n[7] REAL EMBEDDERLLM/QWEN CARRIER GENERATION")
    pipeline_result = _run_common_cover_pipeline(
        topic=topic,
        secret=secret,
        encryption_key=dk1,
        position_key=dk2,
        initial_story="",
        key_label="ECDHE",
        encrypted_packet=encrypted,
        target_positions=positions,
        associated_data=associated_data,
    )
    recovered_plaintext = pipeline_result["recovered_plaintext"]
    assert isinstance(recovered_plaintext, bytes)
    embedding_valid = pipeline_result["embedding_verified"] is True
    extraction_valid = pipeline_result["extracted"] == mapped
    inverse_mapping_valid = pipeline_result["recovered_enc"] == encrypted["enc"]
    authentication_valid = recovered_plaintext == plaintext
    end_to_end_valid = pipeline_result["overall_state"] is True

    print("\n[8] END-TO-END STATUS")
    for label, is_valid in (
        ("ECDHE", shared_secret_matches),
        ("PBKDF2", pbkdf2_valid),
        ("DK1", len(dk1) == 32),
        ("DK2", len(dk2) == 32),
        ("AES-256-GCM", aead_valid),
        ("Authentication Tag", len(encrypted["tag"]) == 16),
        ("Payload Mapping", mapping_valid),
        ("SHAKE-128", positions_valid),
        ("Target Positions", positions_valid),
        ("EmbedderLLM", embedding_valid),
        ("Extraction", extraction_valid),
        ("Inverse Mapping", inverse_mapping_valid),
        ("AEAD Verification", authentication_valid),
        ("End-to-End Verification", end_to_end_valid),
    ):
        print(f"{label}: {'PASS' if is_valid else 'FAIL'}")

    elapsed = time.perf_counter() - started
    print("\nPerformance:")
    print(f"{elapsed:.2f} seconds")
    if not end_to_end_valid:
        raise RuntimeError("ECDHE end-to-end verification failed")


def _prompt_for_mode() -> str:
    while True:
        print("Select mode:")
        print("1) Password / PSK")
        print("2) ECDHE")
        choice = input("Choice [1/2]: ").strip().lower()
        if choice in {"1", "psk", "password"}:
            return "psk"
        if choice in {"2", "ecdhe"}:
            return "ecdhe"
        print("Please choose 1 (PSK) or 2 (ECDHE).")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-SHIELD main prototype demo")
    parser.add_argument("--mode", choices=["psk", "ecdhe"], help="Run in PSK or ECDHE mode")
    parser.add_argument("--topic", default="", help="Story topic for generation")
    parser.add_argument("--secret", default="", help="Secret message to hide")
    parser.add_argument("--password", default="", help="Password for PSK mode")
    parser.add_argument(
        "--crypto-only",
        action="store_true",
        help="Verify ECDHE crypto, mapping, and positions without running the LLM",
    )
    args = parser.parse_args()

    if args.mode is None:
        mode = _prompt_for_mode()
    else:
        mode = args.mode

    if args.crypto_only and mode != "ecdhe":
        parser.error("--crypto-only can only be used with --mode ecdhe")

    while True:
        topic = args.topic.strip() if args.topic else input("Enter topic: ").strip()
        if topic:
            break
        print("Topic cannot be empty.")

    while True:
        secret = args.secret.strip() if args.secret else input("Enter secret message: ").strip()
        if secret:
            break
        print("Secret message cannot be empty.")

    if mode == "psk":
        password = args.password if args.password else getpass.getpass("Enter password: ")
        _run_psk_mode(topic, secret, password)
    elif args.crypto_only:
        _run_ecdhe_crypto_only(topic, secret)
    else:
        _run_ecdhe_mode(topic, secret)

    print("\nMAIN DEMO: COMPLETE")


if __name__ == "__main__":
    main()