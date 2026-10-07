import socket
import threading

import pytest

from app.tcp_protocol import (
    authenticate_positions,
    key_confirmation,
    payload_associated_data,
    receive_json,
    send_json,
)


def test_json_frames_round_trip_across_tcp_socket():
    sender, receiver = socket.socketpair()
    message = {"type": "covert_payload", "cover_text": "A quiet story: café"}
    errors = []

    def send() -> None:
        try:
            send_json(sender, message)
        except Exception as error:
            errors.append(error)
        finally:
            sender.close()

    thread = threading.Thread(target=send)
    thread.start()
    try:
        assert receive_json(receiver) == message
    finally:
        receiver.close()
        thread.join()

    assert not errors


def test_key_confirmation_is_role_and_peer_key_bound():
    secret = b"s" * 32
    public_a = b"a" * 32
    public_b = b"b" * 32

    confirmation = key_confirmation(secret, "server_a", public_a, public_b)
    assert confirmation == key_confirmation(secret, "server_a", public_a, public_b)
    assert confirmation != key_confirmation(secret, "server_b", public_a, public_b)
    assert confirmation != key_confirmation(secret, "server_a", public_b, public_a)
    assert confirmation != key_confirmation(b"x" * 32, "server_a", public_a, public_b)


@pytest.mark.parametrize("length", [0, 3, 8194])
def test_payload_associated_data_rejects_invalid_mapped_length(length):
    with pytest.raises(ValueError):
        payload_associated_data(length)


def test_payload_associated_data_binds_length():
    assert payload_associated_data(36) != payload_associated_data(38)


def test_adaptive_target_positions_are_authenticated():
    dk2 = b"k" * 32
    positions = [89, 1234, 1270]
    mac = authenticate_positions(dk2, positions)

    assert mac == authenticate_positions(dk2, positions)
    assert mac != authenticate_positions(dk2, [89, 1235, 1271])
    with pytest.raises(ValueError):
        authenticate_positions(dk2, [89, 89])
