import pytest

from app.crypto.key_derivation import derive_keys, generate_salt


def test_same_password_same_salt_produces_same_keys():
    password = "correct horse battery staple"
    salt = b"\x01\x02\x03\x04\x05\x06\x07\x08"

    dk1_a, dk2_a = derive_keys(password, salt)
    dk1_b, dk2_b = derive_keys(password, salt)

    assert dk1_a == dk1_b
    assert dk2_a == dk2_b


def test_different_password_same_salt_produces_different_keys():
    password_a = "correct horse battery staple"
    password_b = "correct horse battery staple!"
    salt = b"\x01\x02\x03\x04\x05\x06\x07\x08"

    dk1_a, dk2_a = derive_keys(password_a, salt)
    dk1_b, dk2_b = derive_keys(password_b, salt)

    assert dk1_a != dk1_b
    assert dk2_a != dk2_b


def test_same_password_different_salt_produces_different_keys():
    password = "correct horse battery staple"
    salt_a = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    salt_b = b"\x10\x11\x12\x13\x14\x15\x16\x17"

    dk1_a, dk2_a = derive_keys(password, salt_a)
    dk1_b, dk2_b = derive_keys(password, salt_b)

    assert dk1_a != dk1_b
    assert dk2_a != dk2_b


def test_derived_keys_are_32_bytes_each():
    password = "correct horse battery staple"
    salt = generate_salt()

    dk1, dk2 = derive_keys(password, salt)

    assert len(dk1) == 32
    assert len(dk2) == 32


def test_derived_keys_are_distinct():
    password = "correct horse battery staple"
    salt = generate_salt()

    dk1, dk2 = derive_keys(password, salt)

    assert dk1 != dk2


def test_rejects_invalid_password_or_salt_inputs():
    salt = b"\x01\x02\x03\x04"

    with pytest.raises((TypeError, ValueError)):
        derive_keys("", salt)

    with pytest.raises((TypeError, ValueError)):
        derive_keys("password", b"")

    with pytest.raises(TypeError):
        derive_keys(None, salt)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        derive_keys("password", None)  # type: ignore[arg-type]
