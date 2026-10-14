"""Server B: receive and recover PSK- or TCP-ECDHE-protected covert messages."""

from __future__ import annotations

import argparse
import getpass
import hmac
import os
import socket
from collections.abc import Sequence
from hashlib import sha256
from hmac import new as hmac_new
from sys import stderr
from threading import Event

from app.crypto.aead import AEADError, decrypt
from app.crypto.aead_mapping import character_sequence_to_aead
from app.crypto.ecdhe import derive_ecdhe_keys, derive_shared_secret, generate_key_pair
from app.crypto.ecdhe_carrier import (
    PUBLIC_KEY_EXCHANGE_MESSAGE_COUNT,
    assemble_public_key,
    derive_carrier_position_keys,
    embed_public_key_segment,
    extract_public_key_segment,
    split_public_key,
)
from app.crypto.key_derivation import derive_keys
from app.crypto.mapping import CharacterMap
from app.extraction.extractor import Extractor
from app.tcp_protocol import (
    authenticate_positions,
    decode_hex_field,
    key_confirmation,
    payload_associated_data,
    receive_json,
    send_json,
)


def _print_section(number: int, title: str) -> None:
    print()
    print("=" * 72)
    print(f"[{number}] {title}")
    print("=" * 72, flush=True)


def receive_message(
    host: str,
    port: int,
    *,
    password: str | None = None,
    llm_generator=None,
    key_embedder=None,
    demonstration: bool = False,
    listener_ready: Event | None = None,
    stop_event: Event | None = None,
    bound_address: list[tuple[str, int]] | None = None,
) -> None:
    """Accept one sender and display its extraction/decryption pipeline."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(1)
        bound_host, bound_port = listener.getsockname()[:2]
        print("LLM-SHIELD | SERVER B | RECEIVER", flush=True)
        print(f"\nListening on {bound_host}:{bound_port} ...", flush=True)
        if bound_address is not None:
            bound_address.append((str(bound_host), int(bound_port)))
        if stop_event is not None:
            listener.settimeout(0.25)
        if listener_ready is not None:
            listener_ready.set()
        while True:
            try:
                connection, peer_address = listener.accept()
                break
            except socket.timeout:
                if stop_event is not None and stop_event.is_set():
                    return
    if listener_ready is not None:
        listener_ready.set()

    with connection:
        connection.settimeout(600)
        print(f"\nAccepted TCP connection from {peer_address[0]}:{peer_address[1]}", flush=True)
        hello = receive_json(connection)
        if hello.get("type") != "hello":
            raise ValueError("expected session hello from Server A")
        mode = hello.get("mode")
        topic = hello.get("topic")
        if mode not in {"psk", "ecdhe"}:
            raise ValueError("session mode must be 'psk' or 'ecdhe'")
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("session topic must be a non-empty string")
        _print_section(1, "INPUT")
        print(f"  Mode  : {str(mode).upper()}")
        print(f"  Topic : {topic}")
        if mode == "psk":
            print("  PSK password: <hidden>")

        if mode == "ecdhe":
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
            session_id = decode_hex_field(hello, "session_id", 16)
            carrier_dk1, carrier_dk2 = derive_carrier_position_keys(
                os.environ.get("LLM_SHIELD_PREVIOUS_SHARED_SECRET", ""),
                session_id,
            )
            own_segments = split_public_key(public_key)
            print(f"Server B public key ({len(public_key)} bytes): {public_key.hex()}")
            peer_segments: list[bytes] = []
            for segment_index, own_segment in enumerate(own_segments):
                peer_frame = receive_json(connection)
                peer_segment = extract_public_key_segment(
                    peer_frame,
                    expected_role="server_a",
                    expected_segment_index=segment_index,
                    position_key=carrier_dk1,
                )
                peer_segments.append(peer_segment)
                print(
                    f"Carrier {segment_index * 2 + 1}/"
                    f"{PUBLIC_KEY_EXCHANGE_MESSAGE_COUNT} received (Server A)"
                )

                def report_key_progress(
                    completed: int,
                    total: int,
                    status: str,
                ) -> None:
                    if not status.startswith("LLM inference"):
                        return
                    progress = (
                        f"Embedding Server B key segment "
                        f"{segment_index + 1}/5: {completed}/{total} characters "
                        f"({status})"
                    )
                    print(
                        f"\r{progress:<90}",
                        end="",
                        flush=True,
                    )

                own_frame = embed_public_key_segment(
                    own_segment,
                    role="server_b",
                    segment_index=segment_index,
                    topic=topic,
                    position_key=carrier_dk2,
                    embedder=key_embedder,
                    verbose=demonstration,
                    progress_callback=report_key_progress,
                )
                print()
                send_json(connection, own_frame)
                print(
                    f"Carrier {segment_index * 2 + 2}/"
                    f"{PUBLIC_KEY_EXCHANGE_MESSAGE_COUNT} sent (Server B)"
                )

            peer_public_key = assemble_public_key(peer_segments)
            shared_secret = derive_shared_secret(private_key, peer_public_key)
            peer_confirmation = receive_json(connection)
            peer_mac = peer_confirmation.get("mac")
            expected_a_confirmation = key_confirmation(
                shared_secret, "server_a", peer_public_key, public_key
            )
            if (
                peer_confirmation.get("type") != "key_confirmation"
                or peer_confirmation.get("role") != "server_a"
                or not isinstance(peer_mac, str)
                or not hmac.compare_digest(peer_mac, expected_a_confirmation)
            ):
                raise ValueError("Server A key confirmation failed")

            send_json(
                connection,
                {
                    "type": "key_confirmation",
                    "role": "server_b",
                    "mac": key_confirmation(
                        shared_secret, "server_b", peer_public_key, public_key
                    ),
                },
            )
            print(
                f"ECDHE key exchange complete "
                f"({PUBLIC_KEY_EXCHANGE_MESSAGE_COUNT} alternating carriers)"
            )
            _print_section(2, "KEY ESTABLISHMENT")
            print(f"  Server A public key ({len(peer_public_key)} bytes): {peer_public_key.hex()}")
            print("  X25519 shared-secret agreement and peer confirmation: PASS")
        else:
            password = (
                password
                if password is not None
                else getpass.getpass("Enter shared PSK password: ")
            )
            if not password:
                raise ValueError("PSK password must not be empty")
            shared_secret = b""
            _print_section(2, "KEY ESTABLISHMENT")
            print("  Mode         : Pre-shared key (PSK)")
            print("  Password     : <hidden>")

        salt_message = receive_json(connection)
        if salt_message.get("type") != "kdf_salt":
            raise ValueError("expected PBKDF2 salt from Server A")
        salt = decode_hex_field(salt_message, "salt", 16)
        if mode == "ecdhe":
            dk1, dk2 = derive_ecdhe_keys(shared_secret, salt)
        else:
            dk1, dk2 = derive_keys(password, salt)
        send_json(connection, {"type": "kdf_ready"})
        from app.config import DEFAULT_PBKDF2_CONFIG, get_pbkdf2_iterations

        _print_section(3, "PBKDF2 KEY DERIVATION")
        print(f"  Algorithm    : PBKDF2-HMAC-{DEFAULT_PBKDF2_CONFIG.hash_name.upper()}")
        print(f"  Iterations   : {get_pbkdf2_iterations():,}")
        print(f"  Salt ({len(salt)} bytes): {salt.hex()}")
        print(f"  DK1 length   : {len(dk1)} bytes")
        print(f"  DK2 length   : {len(dk2)} bytes")

        payload = receive_json(connection)
        print("Encrypted carrier received; extracting payload...")
        if payload.get("type") != "covert_payload" or payload.get("mode") != mode:
            raise ValueError("received covert payload mode does not match session")
        mapped_length = payload.get("mapped_length")
        if type(mapped_length) is not int:
            raise ValueError("mapped payload length must be an integer")
        associated_data = payload_associated_data(mapped_length)
        nonce = decode_hex_field(payload, "nonce", 12)
        cover_text = payload.get("cover_text")
        if not isinstance(cover_text, str) or not cover_text:
            raise ValueError("cover text must be a non-empty string")
        _print_section(5, "PAYLOAD MAPPING")
        print(f"  Expected mapped payload: {mapped_length} h4 characters")
        print(f"  Payload representation : AEAD tag || ciphertext ({mapped_length // 2} bytes)")
        supplied_positions = payload.get("target_positions")
        positions_mac = payload.get("positions_mac")

        if (
            not isinstance(supplied_positions, list)
            or len(supplied_positions) != mapped_length
            or any(type(position) is not int for position in supplied_positions)
        ):
            raise ValueError("payload must include one integer target per mapped character")
        positions = supplied_positions
        if (
            not isinstance(positions_mac, str)
            or not hmac.compare_digest(
                positions_mac,
                authenticate_positions(dk2, positions),
            )
        ):
            raise ValueError("target-position authentication failed")
        maximum_position = max(100_000, 32 + mapped_length * 64)
        if positions[-1] >= maximum_position:
            raise ValueError("authenticated target position exceeds the story limit")
        positions_valid = (
            len(positions) == mapped_length
            and all(left < right for left, right in zip(positions, positions[1:]))
        )
        if not positions_valid:
            raise RuntimeError("regenerated SHAKE-128 positions are invalid")
        _print_section(6, "SHAKE-128 POSITION GENERATION")
        print(f"  Position key   : DK2 ({len(dk2)} bytes)")
        print(f"  Target count   : {len(positions)}")
        print(f"  Authenticated final positions: {positions}")
        print("  Positions MAC verification: PASS")

        _print_section(7, "RECEIVED EMBEDDERLLM / QWEN CARRIER")
        print(f"  Carrier length : {len(cover_text)} characters")
        print("  Received carrier text:")
        print(cover_text, flush=True)

        extracted = Extractor().extract(
            cover_text=cover_text,
            positions=positions,
        ).upper()
        if len(extracted) != mapped_length:
            raise ValueError("cover text did not contain all embedded characters")
        _print_section(8, "EXTRACTION")
        print(f"  Extracted characters ({len(extracted)}): {extracted}")
        print("  Extraction: PASS")

        recovered_enc = character_sequence_to_aead(extracted)
        if len(recovered_enc) != mapped_length // 2:
            raise ValueError("inverse-mapped AEAD payload length is inconsistent")
        _print_section(9, "INVERSE MAPPING")
        print(f"  Recovered AEAD payload ({len(recovered_enc)} bytes): {recovered_enc.hex()}")

        recovered_tag = recovered_enc[:16]
        ciphertext = recovered_enc[16:]
        print(f"  Ciphertext ({len(ciphertext)} bytes): {ciphertext.hex()}")
        print(f"  Authentication tag ({len(recovered_tag)} bytes): {recovered_tag.hex()}")
        try:
            recovered_plaintext = decrypt(
                ciphertext=ciphertext,
                tag=recovered_tag,
                nonce=nonce,
                dk1=dk1,
                associated_data=associated_data,
            )
        except AEADError:
            _print_section(10, "AEAD DECRYPTION")
            print("  AES-256-GCM authentication: FAIL", flush=True)
            raise
        recovered_message = recovered_plaintext.decode("utf-8")
        _print_section(10, "AEAD DECRYPTION")
        print("  AES-256-GCM authentication: PASS")
        print(f"  Recovered plaintext       : {recovered_message}")
        send_json(
            connection,
            {
                "type": "delivery_ack",
                "mac": hmac_new(
                    dk1,
                    b"llm-shield/tcp-ecdhe/delivery-ack/v1"
                    + sha256(recovered_plaintext).digest(),
                    "sha256",
                ).hexdigest(),
            },
        )
        _print_section(11, "END-TO-END VERIFICATION")
        print("  Receiver authenticated the recovered message: PASS")
        print("  Authenticated delivery acknowledgement sent to Server A: PASS")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="local bind address")
    parser.add_argument("--port", type=int, default=50505, help="TCP listen port")
    args = parser.parse_args(argv)
    try:
        receive_message(args.host, args.port, demonstration=True)
    except (ConnectionError, OSError, TimeoutError, ValueError) as error:
        print(f"Session failed: {error}", file=stderr, flush=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
