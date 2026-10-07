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

_PSK_AAD = b"llm-shield/psk/password-mode/aes-256-gcm/v1"


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
    total_started: float | None = None,
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
            result = super()._embed_one_character(*args, **kwargs)
            self._demo_completed = getattr(self, "_demo_completed", 0) + 1
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
    if key_label == "PSK":
        mapping_valid = character_sequence_to_aead(mapped) == enc
        print("\n[3] AES-256-GCM / AEAD")
        print("\nPlaintext:")
        print(secret)
        print("\nAAD:")
        effective_aad = associated_data or b""
        print(
            effective_aad.hex()
            if effective_aad
            else "b'' (empty AAD; hex: <empty>)"
        )
        print("\nNonce:")
        print(nonce.hex())
        print("\nCiphertext:")
        print(encrypted["ciphertext"].hex())
        print("\nAuthentication Tag:")
        print(encrypted["tag"].hex())
        print("\nCiphertext Length:")
        print(f"{len(encrypted['ciphertext'])} bytes")
        print("\nAuthentication Tag Length:")
        print(f"{len(encrypted['tag'])} bytes")
        print("\nAEAD Encryption:")
        print(
            "PASS"
            if len(nonce) == 12 and len(encrypted["tag"]) == 16
            and enc == encrypted["tag"] + encrypted["ciphertext"]
            else "FAIL"
        )

        print("\n[4] PAYLOAD MAPPING")
        print("\nOriginal AEAD Payload:")
        print(enc.hex())
        print("\nMapped Payload:")
        print(mapped)
        print("\nMapped Payload Length:")
        print(len(mapped))
        print("\nMapping:")
        print("PASS" if mapping_valid else "FAIL")
        if not mapping_valid:
            raise RuntimeError("PSK AEAD payload mapping did not round-trip")

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
    if key_label == "PSK":
        positions_valid = (
            len(positions) == len(mapped)
            and all(left < right for left, right in zip(positions, positions[1:]))
        )
        print("\n[5] SHAKE-128 POSITION GENERATION")
        print("\nPosition Key:")
        print("DK2")
        print("\nSHAKE-128:")
        print("ACTIVE")
        print("\nNumber of Target Positions:")
        print(len(positions))
        print("\nTarget Position Sequence:")
        print(positions)
        print("\nPositions Strictly Increasing:")
        print("PASS" if positions_valid else "FAIL")
        print("\nPosition Generation:")
        print("PASS" if positions_valid else "FAIL")
        if not positions_valid:
            raise RuntimeError("PSK target positions are invalid")

    safe_initial_story = _build_safe_initial_story(
        initial_story.strip() if initial_story and initial_story.strip() else topic,
        positions,
    )

    generator = LLMGenerator()
    character_map = CharacterMap()
    embedder = DemoEmbedder(llm_generator=generator, character_map=character_map)
    embedder._demo_total = len(mapped)
    if key_label in {"ECDHE", "PSK"}:
        if key_label == "PSK":
            print("\n[6] REAL EMBEDDERLLM / QWEN CARRIER GENERATION")
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

    if key_label in {"ECDHE", "PSK"}:
        if key_label == "PSK":
            print("\n[7] GENERATED COVER TEXT")
        print("\nGenerated Cover Text:")
        print(result.story)
        print("\nEmbedding Positions:")
        print(positions)
        print("\nEmbedded Characters:")
        print(result.embedded_characters)
        if key_label == "PSK":
            print("\n[8] EMBEDDING VERIFICATION")
        print("\nEmbedding Verification:")
        print("PASS" if embedding_verified else "FAIL")
        print("\nCover Naturalness:")
        for check, passed in naturalness_state.items():
            label = check.replace("_", " ").title()
            print(f"{label}: {'PASS' if passed else 'FAIL'}")
        if key_label == "PSK":
            print("\n[9] EXTRACTION")
        print("\nExtracting payload from carrier text...")
        print("\nExtracted Characters:")
        print(canonical_extracted)
        print("\nExtracted Payload Length:")
        print(len(canonical_extracted))
        print("\nExtraction:")
        print("PASS" if canonical_extracted == mapped else "FAIL")
        if key_label == "PSK":
            print("\n[10] INVERSE MAPPING")
        print("\nMapped Payload:")
        print(mapped)
        print("\nRecovered AEAD Payload:")
        print(recovered_enc.hex())
        print("\nInverse Mapping:")
        print("PASS" if recovered_enc == enc else "FAIL")
        print("\nPayload Recovery:")
        print("PASS" if payload_state else "FAIL")
        if key_label == "PSK":
            print("\n[11] AEAD VERIFICATION / DECRYPTION")
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
        if key_label == "PSK":
            print("\n[12] FINAL END-TO-END VERIFICATION")
            print(
                "Original message == recovered message:",
                "PASS" if recovered_plaintext == plaintext else "FAIL",
            )
            print("Payload recovery:", "PASS" if payload_state else "FAIL")
            print(
                "End-to-end verification:",
                "PASS" if overall_state else "FAIL",
            )
            runtime = time.perf_counter() - (
                total_started if total_started is not None else start
            )
            print(f"Total runtime: {runtime:.2f} seconds")
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
        failed_checks = []
        if not embedding_verified:
            failed_checks.append("embedding verification")
        if not payload_state:
            failed_checks.append("payload recovery")
        failed_checks.extend(
            f"cover naturalness: {check}"
            for check, passed in naturalness_state.items()
            if not passed
        )
        raise RuntimeError(
            f"[{key_label}] secure covert pipeline failed: "
            f"{'; '.join(failed_checks)}"
        )

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
    from server_a import send_message

    send_message("127.0.0.1", 50505, secret, topic, mode="psk", password=password)


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
    from server_a import send_message

    send_message("127.0.0.1", 50505, secret, topic, mode="ecdhe")


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
    parser.add_argument("--host", default="127.0.0.1", help="Server B address")
    parser.add_argument("--port", type=int, default=50505, help="Server B TCP port")
    args = parser.parse_args()

    if args.mode is None:
        mode = _prompt_for_mode()
    else:
        mode = args.mode

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
        from server_a import send_message

        send_message(
            args.host, args.port, secret, topic, mode="psk", password=password
        )
    else:
        from server_a import send_message

        send_message(args.host, args.port, secret, topic, mode="ecdhe")


if __name__ == "__main__":
    main()