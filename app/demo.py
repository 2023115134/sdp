"""Interactive Phase-2 secure demo for the project.

This demo uses the existing project modules without monkeypatching the LLM or
crypto stack.

Flow:
    plaintext
      -> PBKDF2
      -> dk1 / dk2
      -> AES-256-GCM using dk1
      -> Enc = Tag || Ciphertext
      -> h4 mapping
      -> dk2-based SHAKE128 positions
      -> existing EmbedderLLM/Qwen
      -> generated story
      -> extraction
      -> reverse mapping
      -> AEAD decrypt using original nonce + dk1
      -> recovered plaintext
"""

from __future__ import annotations

import getpass
import logging
import time

from app.crypto.aead import decrypt, encrypt
from app.crypto.aead_mapping import aead_to_character_sequence, character_sequence_to_aead
from app.crypto.key_derivation import derive_keys, generate_salt
from app.crypto.mapping import CharacterMap
from app.crypto.position_generator import generate_positions
from app.extraction.extractor import Extractor
from app.llm.embedder import EmbedderLLM
from app.llm.generator import LLMGenerator


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)


class DemoEmbedder(EmbedderLLM):
    """Add demo-only timing around the unchanged real embedder operation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._demo_embedding_started = time.perf_counter()

    def _embed_one_character(self, *args, **kwargs):
        character = kwargs.get("character", args[1] if len(args) > 1 else "")
        position = kwargs.get("position", args[2] if len(args) > 2 else 0)
        max_retries = kwargs.get("max_retries", args[6] if len(args) > 6 else 0)
        elapsed = time.perf_counter() - self._demo_embedding_started
        print(
            f"[embedding] character={character!r} position={position} "
            f"retries<= {max_retries} elapsed={elapsed:.1f}s"
        )
        result = super()._embed_one_character(*args, **kwargs)
        self._demo_completed = getattr(self, "_demo_completed", 0) + 1
        total = getattr(self, "_demo_total", "?")
        elapsed = time.perf_counter() - self._demo_embedding_started
        print(
            f"[embedding] progress={self._demo_completed}/{total} "
            f"elapsed={elapsed:.1f}s"
        )
        return result


def main() -> None:
    print("=" * 74)
    print("PHASE-2 SECURE DEMO")
    print("=" * 74)

    while True:
        topic = input("Enter topic : ").strip() 
        if topic:
            break
        print("Topic cannot be empty.")

    secret = input("Enter secret message : ").strip() 

    password = getpass.getpass("Enter password : ") 

    salt = generate_salt()
    dk1, dk2 = derive_keys(password, salt)

    print("\n[0/8] Input and key material")
    print("[0/8] Plaintext:", repr(secret))
    print("[0/8] Salt:", salt.hex())
    print("[0/8] dk1 length:", len(dk1), "bytes")
    print("[0/8] dk2 length:", len(dk2), "bytes")

    print("\n[1/8] Encrypting plaintext with AES-256-GCM using dk1")
    plaintext = secret.encode("utf-8")
    encrypted = encrypt(plaintext, dk1)
    enc = encrypted["enc"]
    nonce = encrypted["nonce"]

    print("[1/8] Plaintext length:", len(plaintext))
    print("[1/8] Ciphertext length:", len(encrypted["ciphertext"]))
    print("[1/8] Nonce:", nonce.hex())
    print("[1/8] Authentication tag:", encrypted["tag"].hex())
    print("[1/8] Enc length:", len(enc))
    print("[1/8] Nonce length:", len(nonce))
    print("[1/8] AEAD tag length:", len(encrypted["tag"]))

    print("\n[2/8] Mapping AEAD payload into h4 character sequence")
    mapped = aead_to_character_sequence(enc)
    print("[2/8] Mapped payload length:", len(mapped))
    print("[2/8] h4 mapped sequence:", repr(mapped))

    # The Phase-2 SHAKE128 generator adds a variable step per character, and
    # with the default 5-bit chunking the worst-case path can require a much
    # larger cover length than the plaintext length itself. Keep a comfortable
    # bound so the demo doesn't fail on real payloads.
    offset_do = 32
    bit_chunk_size = 5
    max_step = offset_do + ((1 << bit_chunk_size) - 1)
    required_story_length = max(
        5000,
        offset_do + len(mapped) * max_step + 512,
    )

    print("\n[3/8] Generating SHAKE128 positions from dk2")
    positions = generate_positions(
        key_material=dk2,
        number_of_positions=len(mapped),
        offset_do=offset_do,
        max_story_length=required_story_length,
        min_gap=1,
    )
    print("[3/8] Number of embedding positions:", len(positions))
    print("[3/8] SHAKE128 positions:", positions)

    print("\n[4/8] Running real EmbedderLLM/Qwen embedding")
    generator = LLMGenerator()
    character_map = CharacterMap()
    embedder = DemoEmbedder(llm_generator=generator, character_map=character_map)
    embedder._demo_total = len(mapped)
    start = time.perf_counter()
    result = embedder.embed(
        topic=topic,
        characters=mapped,
        positions=positions,
        initial_story="",
        temperature=0.7,
        top_k=20,
        max_new_tokens=32,
        max_attempts=2500,
        max_retries=3,
    )
    embed_seconds = time.perf_counter() - start
    print("[4/8] Embedding runtime:", round(embed_seconds, 2), "seconds")
    print("[4/8] Story length:", len(result.story))

    print("\n[5/8] Generated story")
    print(result.story)

    print("\n[6/8] Extraction and reverse mapping")
    extractor = Extractor(position_generator=None, character_map=character_map)
    extracted = extractor.extract(
        cover_text=result.story,
        positions=positions,
    )
    print("[6/8] Extracted payload:", extracted)
    print("[6/8] Extracted sequence:", repr(extracted))
    recovered_enc = character_sequence_to_aead(extracted)
    print("[6/8] Recovered Enc:", recovered_enc.hex())
    print("[6/8] Enc round-trip:", recovered_enc == enc)

    print("\n[7/8] Receiver-side AEAD decryption")
    tag = recovered_enc[:16]
    ciphertext = recovered_enc[16:]
    recovered_plaintext = decrypt(
        ciphertext=ciphertext,
        tag=tag,
        nonce=nonce,
        dk1=dk1,
    )
    print("[7/8] Recovered plaintext:", recovered_plaintext.decode("utf-8"))

    print("\n[8/8] Verification")
    pass_state = (
        recovered_plaintext == plaintext
        and extracted == mapped
        and recovered_enc == enc
    )
    print("=" * 74)
    print("FINAL RESULT:", "PASS" if pass_state else "FAIL")
    print("=" * 74)

    print("\nInput topic:", topic)
    print("Input message:", secret)
    print("Mapped payload length:", len(mapped))
    print("Embedding positions:", len(positions))
    print("Recovered plaintext:", recovered_plaintext.decode("utf-8"))
    print("Overall status:", "PASS" if pass_state else "FAIL")


if __name__ == "__main__":
    main()