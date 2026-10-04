"""Run an in-process bilateral ECDHE and AES-GCM communication demo."""

from __future__ import annotations

import os

from .aead import decrypt, encrypt
from .ecdhe import derive_session_keys, generate_key_pair


def main() -> None:
    alice_private, alice_public = generate_key_pair()
    bob_private, bob_public = generate_key_pair()
    salt = os.urandom(16)
    context = b"llm-shield/demo-session"

    alice_send, alice_receive = derive_session_keys(
        alice_private, bob_public, salt, context
    )
    bob_send, bob_receive = derive_session_keys(
        bob_private, alice_public, salt, context
    )

    alice_message = b"Hello from Alice"
    alice_packet = encrypt(alice_message, alice_send, context)
    received_by_bob = decrypt(
        alice_packet["ciphertext"],
        alice_packet["tag"],
        alice_packet["nonce"],
        bob_receive,
        context,
    )
    assert received_by_bob == alice_message
    print("Alice -> Bob:", received_by_bob.decode("utf-8"))

    bob_message = b"Hello from Bob"
    bob_packet = encrypt(bob_message, bob_send, context)
    received_by_alice = decrypt(
        bob_packet["ciphertext"],
        bob_packet["tag"],
        bob_packet["nonce"],
        alice_receive,
        context,
    )
    assert received_by_alice == bob_message
    print("Bob -> Alice:", received_by_alice.decode("utf-8"))
    print("ECDHE + AES-GCM in-process demo: PASS")


if __name__ == "__main__":
    main()