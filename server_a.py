"""Server A: select PSK or ECDHE and send a covert message to Server B."""

from __future__ import annotations

import argparse
import getpass
import hmac
import os
import socket
import sys
import time
from collections.abc import Sequence
from hashlib import sha256
from hmac import new as hmac_new

from app.crypto.aead import encrypt
from app.crypto.aead_mapping import (
    aead_to_character_sequence,
    character_sequence_to_aead,
)
from app.crypto.ecdhe import derive_ecdhe_keys, derive_shared_secret, generate_key_pair
from app.crypto.ecdhe_carrier import (
    PUBLIC_KEY_EXCHANGE_MESSAGE_COUNT,
    assemble_public_key,
    derive_carrier_position_keys,
    embed_public_key_segment,
    extract_public_key_segment,
    split_public_key,
)
from app.crypto.key_derivation import derive_keys, generate_salt
from app.crypto.mapping import CharacterMap
from app.tcp_protocol import (
    MAX_MAPPED_CHARACTERS,
    authenticate_positions,
    key_confirmation,
    payload_associated_data,
    positions_for_payload,
    receive_json,
    send_json,
)


def _print_section(number: int, title: str) -> None:
    print()
    print("=" * 72)
    print(f"[{number}] {title}")
    print("=" * 72, flush=True)


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


def send_message(
    host: str,
    port: int,
    message: str,
    topic: str,
    mode: str = "ecdhe",
    password: str | None = None,
    *,
    llm_generator=None,
    key_embedder=None,
    demonstration: bool = False,
) -> None:
    """Send one PSK- or TCP-ECDHE-protected message to Server B."""

    mode = mode.lower()
    if mode not in {"psk", "ecdhe"}:
        raise ValueError("mode must be 'psk' or 'ecdhe'")
    if not isinstance(message, str):
        raise TypeError("message must be a string")
    if not topic.strip():
        raise ValueError("topic must not be empty")
    if mode == "psk" and not password:
        raise ValueError("PSK mode requires a non-empty password")

    if demonstration:
        _print_section(1, "INPUT")
        print(f"  Mode           : {mode.upper()}")
        print(f"  Topic          : {topic}")
        print(f"  Secret message : {message if message else '<prompt after key setup>'}")
        print("  PSK password   : <hidden>" if mode == "psk" else "  PSK password   : not used")
    print(f"Server A starting {mode.upper()} session with {host}:{port}", flush=True)

    private_key: bytes | None = None
    public_key: bytes | None = None
    if mode == "ecdhe" and (llm_generator is None or key_embedder is None):
        private_key, public_key = generate_key_pair()
        from app.llm.embedder import EmbedderLLM
        from app.llm.generator import LLMGenerator

        if llm_generator is None:
            llm_generator = LLMGenerator()
        if key_embedder is None:
            key_embedder = EmbedderLLM(
                llm_generator=llm_generator,
                character_map=CharacterMap(),
            )
    elif mode == "ecdhe":
        private_key, public_key = generate_key_pair()
    else:
        private_key = None
        public_key = None

    try:
        connection = socket.create_connection((host, port), timeout=15)
    except OSError as error:
        raise ConnectionError(
            f"Could not connect to Server B at {host}:{port}; "
            "start server_b.py first and check the configured address and port."
        ) from error

    with connection:
        connection.settimeout(600)
        print(f"TCP connection to Server B ({host}:{port}): PASS", flush=True)
        session_id: bytes | None = None
        carrier_dk1: bytes | None = None
        carrier_dk2: bytes | None = None
        hello_message: dict[str, object] = {
            "type": "hello",
            "mode": mode,
            "topic": topic,
        }
        if mode == "ecdhe":
            session_id = os.urandom(16)
            carrier_dk1, carrier_dk2 = derive_carrier_position_keys(
                os.environ.get("LLM_SHIELD_PREVIOUS_SHARED_SECRET", ""),
                session_id,
            )
            hello_message["session_id"] = session_id.hex()
        send_json(connection, hello_message)

        if mode == "ecdhe":
            assert private_key is not None and public_key is not None
            assert carrier_dk1 is not None and carrier_dk2 is not None
            if demonstration:
                _print_section(2, "KEY ESTABLISHMENT")
                print(f"  Server A public key ({len(public_key)} bytes): {public_key.hex()}")
            own_segments = split_public_key(public_key)
            peer_segments: list[bytes] = []
            for segment_index, own_segment in enumerate(own_segments):
                def report_key_progress(
                    completed: int,
                    total: int,
                    status: str,
                ) -> None:
                    if not status.startswith("LLM inference"):
                        if demonstration:
                            return
                        print(
                            f"\rEmbedding Server A key segment "
                            f"{segment_index + 1}/5: {completed}/{total} ({status})",
                            end="",
                            flush=True,
                        )
                        return
                    print(f"\n  {status}", flush=True)

                own_frame = embed_public_key_segment(
                    own_segment,
                    role="server_a",
                    segment_index=segment_index,
                    topic=topic,
                    position_key=carrier_dk1,
                    embedder=key_embedder,
                    verbose=demonstration,
                    progress_callback=report_key_progress,
                )
                print()
                send_json(connection, own_frame)
                print(
                    f"Carrier {segment_index * 2 + 1}/"
                    f"{PUBLIC_KEY_EXCHANGE_MESSAGE_COUNT} sent (Server A)"
                )

                peer_frame = receive_json(connection)
                peer_segment = extract_public_key_segment(
                    peer_frame,
                    expected_role="server_b",
                    expected_segment_index=segment_index,
                    position_key=carrier_dk2,
                )
                peer_segments.append(peer_segment)
                print(
                    f"Carrier {segment_index * 2 + 2}/"
                    f"{PUBLIC_KEY_EXCHANGE_MESSAGE_COUNT} received (Server B)"
                )

            peer_public_key = assemble_public_key(peer_segments)
            print(
                f"ECDHE key exchange complete "
                f"({PUBLIC_KEY_EXCHANGE_MESSAGE_COUNT} alternating carriers)"
            )

            shared_secret = derive_shared_secret(private_key, peer_public_key)
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
            print("ECDHE public-key exchange and key confirmation: PASS")
            if demonstration:
                print(f"  Server B public key ({len(peer_public_key)} bytes): {peer_public_key.hex()}")
                print("  X25519 shared-secret agreement: PASS")
        else:
            shared_secret = b""
            if demonstration:
                _print_section(2, "KEY ESTABLISHMENT")
                print("  Mode           : Pre-shared key (PSK)")
                print("  Password       : <hidden>")

        salt = generate_salt()
        send_json(connection, {"type": "kdf_salt", "salt": salt.hex()})
        if mode == "ecdhe":
            dk1, dk2 = derive_ecdhe_keys(shared_secret, salt)
        else:
            assert password is not None
            dk1, dk2 = derive_keys(password, salt)
        ready_message = receive_json(connection)
        if ready_message.get("type") != "kdf_ready":
            raise ValueError("Server B did not confirm key derivation")
        print("Session keys derived; ready to protect the message.")
        if demonstration:
            from app.config import DEFAULT_PBKDF2_CONFIG, get_pbkdf2_iterations

            if mode == "psk":
                print("  Both peers supplied matching password-derived key material.")
            _print_section(3, "PBKDF2 KEY DERIVATION")
            print(f"  Algorithm      : PBKDF2-HMAC-{DEFAULT_PBKDF2_CONFIG.hash_name.upper()}")
            print(f"  Iterations     : {get_pbkdf2_iterations():,}")
            print(f"  Salt ({len(salt)} bytes): {salt.hex()}")
            print(f"  DK1 length     : {len(dk1)} bytes")
            print(f"  DK2 length     : {len(dk2)} bytes")

        if not message.strip():
            message = _prompt_nonempty("Enter secret message: ")

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

        if not encryption_valid:
            raise RuntimeError("AES-256-GCM encryption output is invalid")
        if demonstration:
            _print_section(4, "AES-256-GCM / AEAD ENCRYPTION")
        print("AES-256-GCM encryption: PASS")
        print(f"    AAD: {associated_data.hex()}")
        print(f"    Nonce: {encrypted['nonce'].hex()}")
        print(f"    Ciphertext: {encrypted['ciphertext'].hex()}")
        print(f"    Authentication tag: {encrypted['tag'].hex()}")

        mapped_payload = aead_to_character_sequence(encrypted["enc"])
        mapping_valid = len(mapped_payload) == mapped_length
        mapping_valid = (
            mapping_valid
            and character_sequence_to_aead(mapped_payload) == encrypted["enc"]
        )
        if not mapping_valid:
            raise RuntimeError("AEAD payload mapping produced an invalid length")
        if demonstration:
            _print_section(5, "PAYLOAD MAPPING")
            print(f"  AEAD payload   : {encrypted['enc'].hex()}")
            print(f"  AEAD length    : {len(encrypted['enc'])} bytes")
            print(f"  Mapped payload : {mapped_payload}")
            print(f"  Mapped length  : {len(mapped_payload)} h4 characters")
            print("  Mapping round-trip: PASS")
        else:
            print(f"[2] Payload mapped: {len(mapped_payload)} h4 characters")

        positions = positions_for_payload(dk2, len(mapped_payload))
        positions_valid = (
            len(positions) == len(mapped_payload)
            and all(left < right for left, right in zip(positions, positions[1:]))
        )
        if not positions_valid:
            raise RuntimeError("SHAKE-128 target positions are invalid")
        if demonstration:
            _print_section(6, "SHAKE-128 POSITION GENERATION")
            print(f"  Position key    : DK2 ({len(dk2)} bytes)")
            print(f"  Number of targets: {len(positions)}")
            print(f"  Target positions: {positions}")
            print("  Positions strictly increasing: PASS")
        else:
            print(f"[3] SHAKE-128 positions generated: {len(positions)}")

        from app.llm.embedder import EmbedderLLM
        from app.llm.generator import LLMGenerator

        character_map = CharacterMap()
        if llm_generator is None:
            llm_generator = LLMGenerator()
        if key_embedder is None:
            key_embedder = EmbedderLLM(
                llm_generator=llm_generator,
                character_map=character_map,
            )
        embedder = key_embedder
        if demonstration:
            _print_section(7, "EMBEDDERLLM / QWEN EMBEDDING")
            print(f"  Model          : {llm_generator.model_name}")
            print(f"  Payload length : {len(mapped_payload)} characters")
            print("  Embedding ...", flush=True)
        else:
            print(
                f"[4] Loading/generating LLM carrier "
                f"({len(mapped_payload)} characters)...",
                flush=True,
            )

        def report_payload_progress(
            completed: int,
            total: int,
            status: str,
        ) -> None:
            if status.startswith("LLM inference"):
                print(f"\n  {status}", flush=True)
            elif not demonstration:
                progress = (
                    f"LLM embedding: {completed}/{total} characters ({status})"
                )
                print(f"\r{progress:<90}", end="", flush=True)

        embedding_started = time.perf_counter()
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
            verbose=demonstration,
            progress_callback=report_payload_progress if demonstration else None,
        )
        embedding_seconds = time.perf_counter() - embedding_started
        print()
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
        print("LLM embedding verification: PASS")
        if demonstration:
            stats = getattr(embedder, "_embedding_stats", {})
            print("Embedding verification: PASS", flush=True)
            print(f"Generated carrier text ({len(result.story)} characters):")
            print(result.story)
            print(f"Embedded payload    : {mapped_payload}")
            print(f"Final target count  : {len(actual_positions)}")
            print(f"Final target positions: {actual_positions}")
            print(f"Embedding time: {embedding_seconds:.2f} seconds")
            if stats:
                print(f"LLM candidate calls: {stats.get('llm_calls', 0)}")
                print(
                    "Candidate evaluations: "
                    f"{stats.get('candidate_evaluations', 0)}"
                )

        position_authentication = authenticate_positions(dk2, actual_positions)

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
            },
        )
        delivery_ack = receive_json(connection)
        expected_ack = hmac_new(
            dk1,
            b"llm-shield/tcp-ecdhe/delivery-ack/v1" + sha256(plaintext).digest(),
            "sha256",
        ).hexdigest()
        if (
            delivery_ack.get("type") != "delivery_ack"
            or not isinstance(delivery_ack.get("mac"), str)
            or not hmac.compare_digest(delivery_ack["mac"], expected_ack)
        ):
            raise ValueError("Server B did not authenticate successful decryption")
        print("Server B authenticated successful decryption: PASS")
        if demonstration:
            if message != plaintext.decode("utf-8"):
                raise RuntimeError("sender plaintext changed during transmission")
            _print_section(11, "END-TO-END VERIFICATION")
            print("  Receiver authenticated recovered message: PASS")
            print(
                "  Recovered message matches sender input  : "
                "PASS"
            )
            print("  Overall result                         : PASS")
    print(f"{mode.upper()} carrier sent to Server B.")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["psk", "ecdhe"])
    parser.add_argument("--host", default="127.0.0.1", help="Server B address")
    parser.add_argument("--port", type=int, default=50505, help="Server B TCP port")
    parser.add_argument("--topic", default="", help="story topic")
    parser.add_argument("--message", "--secret", dest="message", default="", help="secret message to embed")
    args = parser.parse_args(argv)

    mode = args.mode or _prompt_for_mode()
    topic = args.topic.strip() or _prompt_nonempty("Enter topic: ")
    message = args.message.strip() or _prompt_nonempty("Enter secret message: ")
    password = getpass.getpass("Enter PSK password: ") if mode == "psk" else None

    try:
        send_message(
            host=args.host,
            port=args.port,
            message=message,
            topic=topic,
            mode=mode,
            password=password,
            demonstration=True,
        )
    except (ConnectionError, OSError, TimeoutError) as error:
        print(f"Session failed: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
