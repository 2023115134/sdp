import pytest

from app.crypto.aead import AEADError, decrypt as aead_decrypt, encrypt as aead_encrypt
from app.crypto.ecdhe import decrypt_from_peer, derive_session_keys, encrypt_for_peer, generate_key_pair


def test_ecdhe_round_trip():
    recipient_private, recipient_public = generate_key_pair()

    packet = encrypt_for_peer(b"secret cover payload", recipient_public, b"session-1")

    assert decrypt_from_peer(packet, recipient_private, b"session-1") == b"secret cover payload"
    assert packet["ephemeral_public_key"] != recipient_public


def test_both_peers_derive_directional_keys_and_exchange_messages():
    alice_private, alice_public = generate_key_pair()
    bob_private, bob_public = generate_key_pair()
    salt = b"shared-salt-1234"
    associated_data = b"session-1"

    alice_send, alice_receive = derive_session_keys(
        alice_private, bob_public, salt, associated_data
    )
    bob_send, bob_receive = derive_session_keys(
        bob_private, alice_public, salt, associated_data
    )

    assert alice_send == bob_receive
    assert alice_receive == bob_send
    assert alice_send != alice_receive

    packet = aead_encrypt(b"message from Alice", alice_send, associated_data)
    assert aead_decrypt(
        packet["ciphertext"], packet["tag"], packet["nonce"], bob_receive, associated_data
    ) == b"message from Alice"

    reply = aead_encrypt(b"reply from Bob", bob_send, associated_data)
    assert aead_decrypt(
        reply["ciphertext"], reply["tag"], reply["nonce"], alice_receive, associated_data
    ) == b"reply from Bob"


def test_bilateral_aead_rejects_modified_ciphertext_and_tag():
    alice_private, alice_public = generate_key_pair()
    bob_private, bob_public = generate_key_pair()
    salt = b"shared-salt-1234"
    context = b"session-1"
    alice_send, _ = derive_session_keys(alice_private, bob_public, salt, context)
    _, bob_receive = derive_session_keys(bob_private, alice_public, salt, context)
    packet = aead_encrypt(b"authenticated message", alice_send, context)

    modified_ciphertext = bytearray(packet["ciphertext"])
    modified_ciphertext[0] ^= 1
    with pytest.raises(AEADError):
        aead_decrypt(
            bytes(modified_ciphertext), packet["tag"], packet["nonce"], bob_receive, context
        )

    modified_tag = bytearray(packet["tag"])
    modified_tag[0] ^= 1
    with pytest.raises(AEADError):
        aead_decrypt(
            packet["ciphertext"], bytes(modified_tag), packet["nonce"], bob_receive, context
        )


def test_bilateral_aead_rejects_wrong_direction_and_unrelated_keys():
    alice_private, alice_public = generate_key_pair()
    bob_private, bob_public = generate_key_pair()
    salt = b"shared-salt-1234"
    context = b"session-1"
    alice_send, alice_receive = derive_session_keys(alice_private, bob_public, salt, context)
    bob_send, bob_receive = derive_session_keys(bob_private, alice_public, salt, context)
    packet = aead_encrypt(b"Alice to Bob", alice_send, context)

    with pytest.raises(AEADError):
        aead_decrypt(
            packet["ciphertext"], packet["tag"], packet["nonce"], alice_receive, context
        )
    with pytest.raises(AEADError):
        aead_decrypt(
            packet["ciphertext"], packet["tag"], packet["nonce"], bob_send, context
        )

    assert alice_send == bob_receive
    assert alice_receive == bob_send


def test_bilateral_aead_rejects_mismatched_session_context():
    alice_private, alice_public = generate_key_pair()
    bob_private, bob_public = generate_key_pair()
    salt = b"shared-salt-1234"
    alice_context = b"session-1"
    bob_context = b"session-2"

    alice_send, _ = derive_session_keys(alice_private, bob_public, salt, alice_context)
    _, bob_receive = derive_session_keys(bob_private, alice_public, salt, bob_context)
    packet = aead_encrypt(b"context-bound message", alice_send, alice_context)

    with pytest.raises(AEADError):
        aead_decrypt(
            packet["ciphertext"], packet["tag"], packet["nonce"], bob_receive, alice_context
        )


def test_session_keys_bind_peer_keys_salt_and_associated_data():
    alice_private, alice_public = generate_key_pair()
    bob_private, bob_public = generate_key_pair()
    salt = b"shared-salt-1234"

    expected = derive_session_keys(alice_private, bob_public, salt, b"session-1")
    assert derive_session_keys(alice_private, bob_public, salt, b"session-2") != expected
    assert derive_session_keys(alice_private, bob_public, b"other-salt-12345", b"session-1") != expected
    with pytest.raises(ValueError):
        derive_session_keys(alice_private, alice_public, salt)


def test_each_message_uses_fresh_ephemeral_material():
    recipient_private, recipient_public = generate_key_pair()

    first = encrypt_for_peer(b"same", recipient_public)
    second = encrypt_for_peer(b"same", recipient_public)

    assert first["ephemeral_public_key"] != second["ephemeral_public_key"]
    assert first["salt"] != second["salt"]
    assert decrypt_from_peer(first, recipient_private) == b"same"
    assert decrypt_from_peer(second, recipient_private) == b"same"


def test_wrong_recipient_cannot_decrypt():
    recipient_private, recipient_public = generate_key_pair()
    wrong_private, _ = generate_key_pair()
    packet = encrypt_for_peer(b"recipient only", recipient_public)

    with pytest.raises((AEADError, ValueError)):
        decrypt_from_peer(packet, wrong_private)


def test_associated_data_is_authenticated():
    recipient_private, recipient_public = generate_key_pair()
    packet = encrypt_for_peer(b"bound message", recipient_public, b"header")

    with pytest.raises((AEADError, ValueError)):
        decrypt_from_peer(packet, recipient_private, b"different-header")


def test_ephemeral_public_key_tampering_fails():
    recipient_private, recipient_public = generate_key_pair()
    packet = encrypt_for_peer(b"tamper resistant", recipient_public)
    packet["ephemeral_public_key"] = bytes([packet["ephemeral_public_key"][0] ^ 1]) + packet["ephemeral_public_key"][1:]

    with pytest.raises((AEADError, ValueError)):
        decrypt_from_peer(packet, recipient_private)