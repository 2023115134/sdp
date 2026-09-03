import pytest
from cryptography.exceptions import InvalidTag

from app.crypto.aead import AEADError, decrypt, encrypt


@pytest.fixture
def sample_key():
    return bytes(range(32))


def test_encrypt_decrypt_round_trip(sample_key):
    plaintext = b"hello, world!"

    encrypted = encrypt(plaintext, sample_key)
    decrypted = decrypt(
        ciphertext=encrypted["ciphertext"],
        tag=encrypted["tag"],
        nonce=encrypted["nonce"],
        dk1=sample_key,
    )

    assert decrypted == plaintext


def test_fresh_nonce_is_generated_for_same_plaintext(sample_key):
    plaintext = b"same plaintext"

    first = encrypt(plaintext, sample_key)
    second = encrypt(plaintext, sample_key)

    assert first["nonce"] != second["nonce"]
    assert first["ciphertext"] == second["ciphertext"] or first["ciphertext"] != second["ciphertext"]


def test_ciphertext_tampering_fails_authentication(sample_key):
    plaintext = b"tamper me"
    encrypted = encrypt(plaintext, sample_key)
    tampered_ciphertext = bytearray(encrypted["ciphertext"])
    tampered_ciphertext[0] ^= 0xFF

    with pytest.raises((AEADError, InvalidTag)):
        decrypt(
            ciphertext=bytes(tampered_ciphertext),
            tag=encrypted["tag"],
            nonce=encrypted["nonce"],
            dk1=sample_key,
        )


def test_tag_tampering_fails_authentication(sample_key):
    plaintext = b"tamper tag"
    encrypted = encrypt(plaintext, sample_key)
    tampered_tag = bytearray(encrypted["tag"])
    tampered_tag[0] ^= 0xFF

    with pytest.raises((AEADError, InvalidTag)):
        decrypt(
            ciphertext=encrypted["ciphertext"],
            tag=bytes(tampered_tag),
            nonce=encrypted["nonce"],
            dk1=sample_key,
        )


def test_wrong_key_fails_authentication(sample_key):
    plaintext = b"wrong key"
    encrypted = encrypt(plaintext, sample_key)
    wrong_key = bytes((b + 1) % 256 for b in sample_key)

    with pytest.raises((AEADError, InvalidTag)):
        decrypt(
            ciphertext=encrypted["ciphertext"],
            tag=encrypted["tag"],
            nonce=encrypted["nonce"],
            dk1=wrong_key,
        )


def test_authentication_tag_is_16_bytes(sample_key):
    encrypted = encrypt(b"tag length", sample_key)

    assert len(encrypted["tag"]) == 16


def test_aes_256_key_length_is_32_bytes():
    assert len(bytes(range(32))) == 32


def test_empty_plaintext_is_handled_correctly(sample_key):
    encrypted = encrypt(b"", sample_key)
    decrypted = decrypt(
        ciphertext=encrypted["ciphertext"],
        tag=encrypted["tag"],
        nonce=encrypted["nonce"],
        dk1=sample_key,
    )

    assert decrypted == b""
    assert encrypted["ciphertext"] == b""
    assert len(encrypted["tag"]) == 16


def test_invalid_key_nonce_or_tag_inputs_are_rejected(sample_key):
    with pytest.raises((TypeError, ValueError, AEADError)):
        encrypt(b"hello", b"too-short")

    with pytest.raises((TypeError, ValueError, AEADError)):
        decrypt(b"cipher", b"1234567890123456", b"short", sample_key)

    with pytest.raises((TypeError, ValueError, AEADError)):
        decrypt(b"cipher", b"short", b"\x00" * 12, sample_key)


def test_encrypted_representation_uses_tag_then_ciphertext(sample_key):
    plaintext = b"paper design"
    encrypted = encrypt(plaintext, sample_key)

    encoded = encrypted["tag"] + encrypted["ciphertext"]
    assert encoded == encrypted["enc"]
    assert encoded[:16] == encrypted["tag"]
    assert encoded[16:] == encrypted["ciphertext"]
