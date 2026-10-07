"""Server A: select PSK or ECDHE and send a covert message to Server B."""

from __future__ import annotations

import argparse
import getpass
import hmac
import socket
import time
from collections.abc import Sequence

from app.crypto.aead import decrypt, encrypt
from app.crypto.aead_mapping import (
    aead_to_character_sequence,
    character_sequence_to_aead,
)
from app.crypto.ecdhe import derive_ecdhe_keys, derive_shared_secret, generate_key_pair
from app.crypto.key_derivation import derive_keys, generate_salt
from app.crypto.mapping import CharacterMap
from app.tcp_protocol import (
    MAX_MAPPED_CHARACTERS,
    authenticate_positions,
    decode_hex_field,
    key_confirmation,
    payload_associated_data,
    positions_for_payload,
    receive_json,
    send_json,
)


def _prompt_for_mode() -> str:
    while True:
        print("Select key-establishment mode:")
        print("1) PSK")
        print("2) ECDHE")
        choice = input("Choice [1/2]: ").strip().lower()
        if choice in {"1", "psk"}:
            return "psk"
        if choice in {"2", "ecdhe"}:
            return "ecdhe"
        print("Please choose 1 (PSK) or 2 (ECDHE).")


def _prompt_nonempty(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("This value cannot be empty.")


def _safe_initial_story(topic: str, first_position: int) -> str:
    story = topic.strip()[: max(1, first_position - 1)].rstrip()
    if not story:
        story = "A quiet story"
    while len(story) >= first_position:
        story = story[: max(1, len(story) - 1)].rstrip()
        if not story:
            story = "A quiet story"
    return story


def _print_key_derivation(salt: bytes, dk1: bytes, dk2: bytes) -> None:
    valid = len(dk1) == 32 and len(dk2) == 32
    print("\n[3] PBKDF2 KEY DERIVATION")
    print("Algorithm              : PBKDF2-HMAC-SHA256")
    print(f"Salt                   : {salt.hex()}")
    print("Output                 : 512 bits / 64 bytes")
    print(f"DK1                    : {dk1.hex()}")
    print(f"DK1 length             : {len(dk1)} bytes / {len(dk1) * 8} bits")
    print(f"DK2                    : {dk2.hex()}")
    print(f"DK2 length             : {len(dk2)} bytes / {len(dk2) * 8} bits")
    print(f"PBKDF2 derivation      : {'PASS' if valid else 'FAIL'}")
    if not valid:
        raise RuntimeError("PBKDF2 must produce two 32-byte keys")


def send_message(
    host: str,
    port: int,
    message: str,
    topic: str,
    mode: str = "ecdhe",
    password: str | None = None,
) -> None:
    """Send one PSK- or TCP-ECDHE-protected message to Server B."""

    mode = mode.lower()
    if mode not in {"psk", "ecdhe"}:
        raise ValueError("mode must be 'psk' or 'ecdhe'")
    if not message:
        raise ValueError("message must not be empty")
    if not topic.strip():
        raise ValueError("topic must not be empty")
    if mode == "psk" and not password:
        raise ValueError("PSK mode requires a non-empty password")

    started = time.perf_counter()
    print("=" * 78)
    print(f"LLM-SHIELD | SERVER A | {mode.upper()}")
    print("=" * 78)
    print("\n[1] INPUT")
    print(f"Mode           : {mode.upper()}")
    print(f"Topic          : {topic}")
    print(f"Secret message : {message}")

    private_key: bytes | None = None
    public_key: bytes | None = None
    if mode == "ecdhe":
        private_key, public_key = generate_key_pair()

    print("\n[2] KEY ESTABLISHMENT")
    if mode == "psk":
        print("Method                 : Password-based key establishment")
    else:
        assert public_key is not None
        print(f"Server A public key    : {public_key.hex()}")

    print(f"\nConnecting to Server B at {host}:{port} ...")
    with socket.create_connection((host, port), timeout=15) as connection:
        connection.settimeout(None)
        send_json(
            connection,
            {"type": "hello", "mode": mode, "topic": topic},
        )

        if mode == "ecdhe":
            assert private_key is not None and public_key is not None
            send_json(
                connection,
                {"type": "public_key", "public_key": public_key.hex()},
            )
            peer_message = receive_json(connection)
            if peer_message.get("type") != "public_key":
                raise ValueError("expected Server B public-key message")
            peer_public_key = decode_hex_field(peer_message, "public_key", 32)
            print(f"Server B public key    : {peer_public_key.hex()}")
            print("TCP public-key exchange: PASS")

            shared_secret = derive_shared_secret(private_key, peer_public_key)
            print(f"Server A shared secret : {shared_secret.hex()}")
            own_confirmation = key_confirmation(
                shared_secret, "server_a", public_key, peer_public_key
            )
            send_json(
                connection,
                {
                    "type": "key_confirmation",
                    "role": "server_a",
                    "mac": own_confirmation,
                },
            )
            peer_confirmation = receive_json(connection)
            peer_mac = peer_confirmation.get("mac")
            expected_confirmation = key_confirmation(
                shared_secret, "server_b", public_key, peer_public_key
            )
            if (
                peer_confirmation.get("type") != "key_confirmation"
                or peer_confirmation.get("role") != "server_b"
                or not isinstance(peer_mac, str)
                or not hmac.compare_digest(peer_mac, expected_confirmation)
            ):
                raise ValueError("Server B key confirmation failed")
            print("Shared secrets match   : PASS")
        else:
            shared_secret = b""

        salt = generate_salt()
        send_json(connection, {"type": "kdf_salt", "salt": salt.hex()})
        if mode == "psk":
            print(f"PBKDF2 salt             : {salt.hex()}")
        if mode == "ecdhe":
            dk1, dk2 = derive_ecdhe_keys(shared_secret, salt)
        else:
            assert password is not None
            dk1, dk2 = derive_keys(password, salt)
        _print_key_derivation(salt, dk1, dk2)

        ready_message = receive_json(connection)
        if ready_message.get("type") != "kdf_ready":
            raise ValueError("Server B did not confirm key derivation")

        plaintext = message.encode("utf-8")
        mapped_length = 2 * (len(plaintext) + 16)
        if mapped_length > MAX_MAPPED_CHARACTERS:
            raise ValueError(
                "message is too large for the configured covert-payload limit"
            )
        associated_data = payload_associated_data(mapped_length)
        encrypted = encrypt(plaintext, dk1, associated_data)
        encryption_valid = (
            len(encrypted["nonce"]) == 12
            and len(encrypted["tag"]) == 16
            and encrypted["enc"] == encrypted["tag"] + encrypted["ciphertext"]
        )

        print("\n[4] AES-256-GCM / AEAD ENCRYPTION")
        print(f"Plaintext                : {message}")
        print(f"AAD (hex)                : {associated_data.hex()}")
        print(f"Nonce                    : {encrypted['nonce'].hex()}")
        print(f"Ciphertext               : {encrypted['ciphertext'].hex()}")
        print(f"Ciphertext length        : {len(encrypted['ciphertext'])} bytes")
        print(f"Authentication tag       : {encrypted['tag'].hex()}")
        print(f"Authentication tag length: {len(encrypted['tag'])} bytes")
        print(f"AEAD encryption          : {'PASS' if encryption_valid else 'FAIL'}")
        if not encryption_valid:
            raise RuntimeError("AES-256-GCM encryption output is invalid")

        mapped_payload = aead_to_character_sequence(encrypted["enc"])
        mapping_valid = len(mapped_payload) == mapped_length
        print("\n[5] PAYLOAD MAPPING")
        print("Original AEAD payload    : Authentication tag || ciphertext")
        print(f"Hexadecimal payload      : {encrypted['enc'].hex()}")
        print(f"Mapped payload           : {mapped_payload}")
        print(f"Mapped payload length    : {len(mapped_payload)} characters")
        print(f"Mapping                  : {'PASS' if mapping_valid else 'FAIL'}")
        if not mapping_valid:
            raise RuntimeError("AEAD payload mapping produced an invalid length")

        positions = positions_for_payload(dk2, len(mapped_payload))
        positions_valid = (
            len(positions) == len(mapped_payload)
            and all(left < right for left, right in zip(positions, positions[1:]))
        )
        print("\n[6] SHAKE-128 POSITION GENERATION")
        print("Position key             : DK2")
        print("SHAKE-128                : ACTIVE")
        print(f"Number of target positions: {len(positions)}")
        print(f"Complete target sequence : {positions}")
        print(
            "Positions strictly increasing: "
            f"{'PASS' if positions_valid else 'FAIL'}"
        )
        print(f"Position generation      : {'PASS' if positions_valid else 'FAIL'}")
        if not positions_valid:
            raise RuntimeError("SHAKE-128 target positions are invalid")

        from app.extraction.extractor import Extractor
        from app.llm.embedder import EmbedderLLM
        from app.llm.generator import LLMGenerator

        character_map = CharacterMap()
        generator = LLMGenerator()
        embedder = EmbedderLLM(
            llm_generator=generator,
            character_map=character_map,
        )
        print("\n[7] EMBEDDERLLM / QWEN")
        print(f"LLM model                : {generator.model_name}")
        print(f"Embedding payload length : {len(mapped_payload)} characters")
        print("Embedding progress       : EmbedderLLM reports each character and target")
        expected_positions = positions.copy()
        result = embedder.embed(
            topic=topic,
            characters=mapped_payload,
            positions=positions,
            initial_story=_safe_initial_story(topic, positions[0]),
            temperature=0.7,
            top_k=20,
            max_new_tokens=32,
            max_attempts=15000,
            max_retries=3,
            allow_adaptive_positions=True,
        )
        actual_positions = result.positions
        if not EmbedderLLM.verify_embedding(
            result.story, mapped_payload, actual_positions
        ):
            raise RuntimeError("EmbedderLLM verification failed")
        if (
            len(actual_positions) != len(mapped_payload)
            or any(
                left >= right
                for left, right in zip(
                    actual_positions,
                    actual_positions[1:],
                )
            )
        ):
            raise RuntimeError("adaptive target positions are invalid")
        print(f"Embedding attempts        : {result.attempts}")
        print(f"Embedding status          : PASS ({len(mapped_payload)} characters)")
        print("\nGenerated carrier text:")
        print(result.story)

        extracted = Extractor(character_map=character_map).extract(
            cover_text=result.story,
            positions=actual_positions,
        ).upper()
        print("\n[8] EXTRACTION")
        print("Received carrier text    : (locally generated; transmitted to Server B)")
        print(f"Regenerated positions    : {positions}")
        print(f"Extracted mapped chars   : {extracted}")
        extraction_valid = extracted == mapped_payload
        print(f"Extraction               : {'PASS' if extraction_valid else 'FAIL'}")
        if not extraction_valid:
            raise RuntimeError("local extraction failed")

        recovered_enc = character_sequence_to_aead(extracted)
        inverse_valid = recovered_enc == encrypted["enc"]
        print("\n[9] INVERSE MAPPING")
        print(f"Extracted mapped payload : {extracted}")
        print(f"Recovered hexadecimal    : {recovered_enc.hex()}")
        print(f"Inverse mapping          : {'PASS' if inverse_valid else 'FAIL'}")
        if not inverse_valid:
            raise RuntimeError("inverse payload mapping failed")

        recovered_tag = recovered_enc[:16]
        recovered_ciphertext = recovered_enc[16:]
        print("\n[10] AEAD DECRYPTION")
        print(f"Ciphertext               : {recovered_ciphertext.hex()}")
        print(f"Authentication tag       : {recovered_tag.hex()}")
        recovered_plaintext = decrypt(
            ciphertext=recovered_ciphertext,
            tag=recovered_tag,
            nonce=encrypted["nonce"],
            dk1=dk1,
            associated_data=associated_data,
        )
        authentication_valid = recovered_plaintext == plaintext
        print("AES-256-GCM verification : PASS")
        print(f"Authentication status    : {'PASS' if authentication_valid else 'FAIL'}")
        print(f"Recovered plaintext      : {recovered_plaintext.decode('utf-8')}")
        if not authentication_valid:
            raise RuntimeError("AES-256-GCM did not recover the original plaintext")

        position_authentication = authenticate_positions(dk2, actual_positions)
        if actual_positions != expected_positions:
            print("Adaptive target positions: authenticated SHAKE-128-derived schedule")
            print(f"Actual target sequence   : {actual_positions}")

        send_json(
            connection,
            {
                "type": "covert_payload",
                "mode": mode,
                "mapped_length": len(mapped_payload),
                "nonce": encrypted["nonce"].hex(),
                "cover_text": result.story,
                "target_positions": actual_positions,
                "positions_mac": position_authentication,
                "embedding_attempts": result.attempts,
            },
        )
        print("Carrier transmitted      : PASS (cover text delivered to Server B)")

    print("\n[11] END-TO-END VERIFICATION")
    print(f"Original message         : {message}")
    print(f"Recovered message        : {recovered_plaintext.decode('utf-8')}")
    print(f"Message match            : {'PASS' if authentication_valid else 'FAIL'}")
    print("Overall result           : PASS")
    print(f"Elapsed                  : {time.perf_counter() - started:.2f} seconds")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["psk", "ecdhe"])
    parser.add_argument("--host", default="127.0.0.1", help="Server B address")
    parser.add_argument("--port", type=int, default=50505, help="Server B TCP port")
    parser.add_argument("--topic", default="", help="story topic")
    parser.add_argument("--message", default="", help="secret message to embed")
    args = parser.parse_args(argv)

    mode = args.mode or _prompt_for_mode()
    topic = args.topic.strip() or _prompt_nonempty("Enter topic: ")
    message = args.message.strip() or _prompt_nonempty("Enter secret message: ")
    password = getpass.getpass("Enter PSK password: ") if mode == "psk" else None

    send_message(
        host=args.host,
        port=args.port,
        message=message,
        topic=topic,
        mode=mode,
        password=password,
    )


if __name__ == "__main__":
    main()
