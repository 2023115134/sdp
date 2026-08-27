from app.config import get_model_name
from app.crypto.mapping import CharacterMap


def test_end_to_end_contract_uses_qwen_and_reversible_mapping():
    mapping = CharacterMap()
    payload = mapping.encode("HELLO")
    assert mapping.decode(payload) == "HELLO"
    assert get_model_name() == "Qwen/Qwen2.5-0.5B-Instruct"
