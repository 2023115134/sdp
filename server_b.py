"""Server B: receive and recover PSK- or TCP-ECDHE-protected covert messages."""

from __future__ import annotations

import argparse
import getpass
import hmac
import socket
import time
from collections.abc import Sequence

from app.crypto.aead import decrypt
from app.crypto.aead_mapping import character_sequence_to_aead
from app.crypto.ecdhe import derive_ecdhe_keys, derive_shared_secret, generate_key_pair
from app.crypto.key_derivation import derive_keys
from app.extraction.extractor import Extractor
from app.tcp_protocol import (
    decode_hex_field,
    authenticate_positions,
    key_confirmation,
    payload_associated_data,
    positions_for_payload,
    receive_json,
    send_json,
)


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


def receive_message(host: str, port: int) -> None:
    """Accept one sender and display its extraction/decryption pipeline."""

    started = time.perf_counter()
    print("=" * 78)
    print("LLM-SHIELD | SERVER B | RECEIVER")
    print("=" * 78)
    print(f"Listening on {host}:{port} ...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(1)
        connection, peer_address = listener.accept()

    with connection:
        print(f"TCP connection accepted from {peer_address}")
        hello = receive_json(connection)
        if hello.get("type") != "hello":
            raise ValueError("expected session hello from Server A")
        mode = hello.get("mode")
        topic = hello.get("topic")
        if mode not in {"psk", "ecdhe"}:
            raise ValueError("session mode must be 'psk' or 'ecdhe'")
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("session topic must be a non-empty string")

        print("\n[1] INPUT")
        print(f"Mode           : {mode.upper()}")
        print(f"Topic          : {topic}")
        print("Secret message : encrypted; unavailable until AES-GCM verification")

        print("\n[2] KEY ESTABLISHMENT")
        if mode == "ecdhe":
            private_key, public_key = generate_key_pair()
            peer_message = receive_json(connection)
            if peer_message.get("type") != "public_key":
                raise ValueError("expected Server A public-key message")
            peer_public_key = decode_hex_field(peer_message, "public_key", 32)
            print(f"Server A public key    : {peer_public_key.hex()}")
            print(f"Server B public key    : {public_key.hex()}")
            send_json(
                connection,
                {"type": "public_key", "public_key": public_key.hex()},
            )
            print("TCP public-key exchange: PASS")

            shared_secret = derive_shared_secret(private_key, peer_public_key)
            print(f"Server B shared secret : {shared_secret.hex()}")
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
            print("Shared secrets match   : PASS")
        else:
            print("Method                 : Password-based key establishment")
            password = getpass.getpass("Enter shared PSK password: ")
            if not password:
                raise ValueError("PSK password must not be empty")
            shared_secret = b""

        salt_message = receive_json(connection)
        if salt_message.get("type") != "kdf_salt":
            raise ValueError("expected PBKDF2 salt from Server A")
        salt = decode_hex_field(salt_message, "salt", 16)
        if mode == "psk":
            print(f"PBKDF2 salt             : {salt.hex()}")
        if mode == "ecdhe":
            dk1, dk2 = derive_ecdhe_keys(shared_secret, salt)
        else:
            dk1, dk2 = derive_keys(password, salt)
        _print_key_derivation(salt, dk1, dk2)
        send_json(connection, {"type": "kdf_ready"})

        payload = receive_json(connection)
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
        embedding_attempts = payload.get("embedding_attempts")

        print("\n[4] AES-256-GCM / AEAD ENCRYPTION")
        print("Ciphertext and authentication tag remain hidden in the cover payload.")
        print("\n[5] PAYLOAD MAPPING")
        print("AEAD payload            : authentication tag || ciphertext")
        print(f"Mapped payload length   : {mapped_length} h4 characters")

        supplied_positions = payload.get("target_positions")
        positions_mac = payload.get("positions_mac")

        base_positions = positions_for_payload(dk2, mapped_length)
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
        print("\n[6] SHAKE-128 POSITION GENERATION")
        print("Position key             : DK2")
        print("SHAKE-128                : ACTIVE")
        print(f"Number of target positions: {len(positions)}")
        print(f"SHAKE-128 base sequence  : {base_positions}")
        print(f"Authenticated target sequence: {positions}")
        print(
            "Positions strictly increasing: "
            f"{'PASS' if positions_valid else 'FAIL'}"
        )
        print(f"Position generation      : {'PASS' if positions_valid else 'FAIL'}")
        if not positions_valid:
            raise RuntimeError("regenerated SHAKE-128 positions are invalid")

        print("\n[7] EMBEDDERLLM / QWEN")
        print("LLM model                : generated by Server A")
        print(f"Embedding payload length : {mapped_length} characters")
        print(f"Embedding attempts       : {embedding_attempts}")
        print("Embedding status         : cover received from Server A")
        print("\nReceived carrier text:")
        print(cover_text)

        print("\n[8] EXTRACTION")
        print("Regenerated target positions:")
        print(positions)
        extracted = Extractor().extract(
            cover_text=cover_text,
            positions=positions,
        ).upper()
        if len(extracted) != mapped_length:
            raise ValueError("cover text did not contain all embedded characters")
        print(f"Extracted mapped chars   : {extracted}")
        print("Extraction               : PASS")

        print("\n[9] INVERSE MAPPING")
        recovered_enc = character_sequence_to_aead(extracted)
        if len(recovered_enc) != mapped_length // 2:
            raise ValueError("inverse-mapped AEAD payload length is inconsistent")
        print(f"Extracted mapped payload: {extracted}")
        print(f"Recovered hexadecimal   : {recovered_enc.hex()}")
        print("Inverse mapping         : PASS")

        recovered_tag = recovered_enc[:16]
        ciphertext = recovered_enc[16:]
        print("\n[10] AEAD DECRYPTION")
        print(f"Ciphertext               : {ciphertext.hex()}")
        print(f"Authentication tag       : {recovered_tag.hex()}")
        recovered_plaintext = decrypt(
            ciphertext=ciphertext,
            tag=recovered_tag,
            nonce=nonce,
            dk1=dk1,
            associated_data=associated_data,
        )
        recovered_message = recovered_plaintext.decode("utf-8")
        print("AES-256-GCM verification : PASS")
        print("Authentication status    : PASS")
        print(f"Recovered plaintext      : {recovered_message}")

        print("\n[11] END-TO-END VERIFICATION")
        print("Original message         : hidden from receiver until decrypted")
        print(f"Recovered message        : {recovered_message}")
        print("Message match            : PASS (AEAD authenticated)")
        print("Overall result           : PASS")
        print(f"Elapsed                  : {time.perf_counter() - started:.2f} seconds")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="local bind address")
    parser.add_argument("--port", type=int, default=50505, help="TCP listen port")
    args = parser.parse_args(argv)
    receive_message(args.host, args.port)


if __name__ == "__main__":
    main()
