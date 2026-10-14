from types import SimpleNamespace

import pytest

from app.crypto.ecdhe_carrier import (
    derive_carrier_position_keys,
    embed_public_key_segment,
    extract_public_key_segment,
    split_public_key,
)
from app.llm.embedder import EmbedderLLM


def _fake_embed(self, topic, characters, positions, initial_story="", **kwargs):
    story = list(initial_story)
    if positions and len(story) <= positions[-1]:
        story.extend(" " * (positions[-1] + 1 - len(story)))
    for character, position in zip(characters, positions):
        story[position] = character
    return SimpleNamespace(story="".join(story), positions=list(positions), attempts=1)


def test_public_key_carrier_positions_use_cross_peer_pbkdf2_keys(monkeypatch):
    monkeypatch.setattr(EmbedderLLM, "embed", _fake_embed)
    session_id = b"s" * 16
    dk1, dk2 = derive_carrier_position_keys("", session_id)
    public_key = bytes(range(32))
    segment_a = split_public_key(public_key)[0]
    segment_b = split_public_key(public_key)[1]

    frame_a = embed_public_key_segment(
        segment_a,
        role="server_a",
        segment_index=0,
        topic="A quiet city street at dusk",
        position_key=dk1,
    )
    frame_b = embed_public_key_segment(
        segment_b,
        role="server_b",
        segment_index=1,
        topic="A quiet city street at dusk",
        position_key=dk2,
    )

    assert "positions" not in frame_a
    assert "positions" not in frame_b
    assert extract_public_key_segment(
        frame_a,
        expected_role="server_a",
        expected_segment_index=0,
        position_key=dk1,
    ) == segment_a
    assert extract_public_key_segment(
        frame_b,
        expected_role="server_b",
        expected_segment_index=1,
        position_key=dk2,
    ) == segment_b

    with pytest.raises((ValueError, IndexError)):
        extract_public_key_segment(
            frame_a,
            expected_role="server_a",
            expected_segment_index=0,
            position_key=dk2,
        )


def test_carrier_position_keys_match_only_with_the_same_secret_and_salt():
    session_id = b"s" * 16
    dk1_a, dk2_a = derive_carrier_position_keys("", session_id)
    dk1_b, dk2_b = derive_carrier_position_keys("", session_id)
    other_dk1, other_dk2 = derive_carrier_position_keys("prior secret", session_id)
    changed_salt_dk1, changed_salt_dk2 = derive_carrier_position_keys(
        "", b"t" * 16
    )

    assert (dk1_a, dk2_a) == (dk1_b, dk2_b)
    assert dk1_a != dk2_a
    assert (dk1_a, dk2_a) != (other_dk1, other_dk2)
    assert (dk1_a, dk2_a) != (changed_salt_dk1, changed_salt_dk2)
