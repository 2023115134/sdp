# LLM-SHIELD

This repository contains the Phase 1 implementation of an LLM-based covert communication and steganography system inspired by the research paper specification.

## Project Overview

The project aims to build a research-oriented prototype for covert communication using large language models. The current implementation focuses on the core Phase 1 components:

- LLM generation engine
- reversible character mapping
- cryptographic position generation
- EmbedderLLM prototype
- extraction prototype
- X25519 ECDHE key exchange with HKDF-SHA256 and AES-256-GCM transport encryption
- end-to-end validation experiment

## Current Phase

Phase 1 is the core implementation stage and includes the essential research contribution for the project. This version does not include a frontend, login system, or deployment layer.

## Project Structure

```text
project/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── crypto/
│   │   ├── __init__.py
│   │   ├── mapping.py
│   │   └── position_generator.py
│   ├── extraction/
│   │   ├── __init__.py
│   │   └── extractor.py
│   └── llm/
│       ├── __init__.py
│       ├── embedder.py
│       └── generator.py
├── experiments/
│   └── phase1_experiment.py
├── tests/
│   ├── test_embedding.py
│   ├── test_extraction.py
│   ├── test_mapping.py
│   └── test_positions.py
├── .gitignore
├── IMPLEMENTATION_NOTES.md
├── README.md
├── requirements.txt
└── ...
```

## Requirements

Before running the project, the laptop should have:

- Windows with PowerShell
- Python 3.11 installed and available as `python`
- Internet access for installing packages and downloading the default Hugging Face model
- At least 8 GB RAM and approximately 5 GB of free disk space for the Python packages, model files, and cache
- A CPU is sufficient; an NVIDIA GPU is optional. When a CUDA-enabled PyTorch build is installed, the generator selects CUDA automatically.

The complete Python package list is maintained in [requirements.txt](requirements.txt). Check that file before installation and install it with the command below.

## Step-by-step setup

Run the following commands in **PowerShell** from the project folder.

### 1. Open the project folder

```powershell
cd "C:\Users\MY PC\llm-shield\project"
```

### 2. Create the virtual environment

Run this only if `.venv` does not already exist:

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

````powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

```.\.venv\Scripts\Activate.ps1

The prompt should begin with `(.venv)` after activation.

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
````

The default model is `Qwen/Qwen2.5-0.5B-Instruct`. Hugging Face may download it the first time a model-backed command runs. Keep the terminal connected to the internet for that first run.

Story generation is unseeded by default, so repeated runs with the same topic may produce different Qwen cover text. Mapping and SHAKE128 position generation remain deterministic. Pass an explicit `seed` to `LLMGenerator` when reproducible model output is needed; `deterministic=True` is also available for the generator test path.

## Run the tests

Run the complete automated test suite:

```powershell
python -m pytest -q
```

Expected result:

```text
16 passed
```

## Run the paper-based workflow

Run these commands in this order while the `(.venv)` environment is active.

### Stage 1: Test the Qwen generator

```powershell
python -m app.llm.generator
```

This loads Qwen, generates a short continuation, prints top-k candidates, and verifies that their probabilities are sorted.

### Stage 2: Test one-character embedding

```powershell
python -m app.llm.embedder
```

This uses one hidden character, `E`, at position `50`. It has bounded retries and candidate evaluations for laptop safety. Candidate selection keeps only tokens that satisfy the exact fixed embedding position, then ranks them by model log-probability with local naturalness and context signals.

### Stage 3: Test extraction

```powershell
python -m app.extraction.extractor
```

This independently extracts and decodes the known `HELLO` payload.

### Stage 4: Run the complete demo

```powershell
python -m app.demo
```

The demo prompts for `Enter secret message:` and `Enter topic:`. The secret may contain any non-empty text. A blank topic is rejected and prompts again. The entered topic is used as the initial story context for Qwen and `EmbedderLLM`; no topic or secret is hardcoded.

The official demo uses fixed paper mode: positions are generated before embedding with `PositionGenerator.generate_for_message()` and the `test-secret-key` configuration. For example, `hi` maps to `IRIH` and produces deterministic positions `[68, 129, 185, 223]`. The optional `EmbedderLLM.embed_dynamic()` method remains available for experimental comparisons but is not used by the official demo.

The demo performs:

```text
secret message
	-> h4 character mapping
	-> deterministic SHAKE128 position generation
	-> fixed target positions
	-> Qwen candidate-token embedding
	-> cover-text validation
	-> position-based extraction
	-> decoding
	-> wrong-key validation
```

The final successful line is:

```text
END-TO-END TEST: PASS
```

The demo automatically uses the best available backend. Set `LLM_DEVICE=cpu` to force CPU execution, or `LLM_DEVICE=cuda` / `LLM_DEVICE=cuda:0` to request a specific CUDA device. If CUDA is requested but unavailable, the generator logs a warning and falls back to CPU. Longer secrets and fixed-position candidate searches can take time. Do not start multiple demo processes at the same time.

The active backend is logged when the model loads. To use an NVIDIA GPU, install a CUDA-enabled PyTorch wheel for the selected Python environment; `torch>=2.2.0` alone may resolve to a CPU-only wheel on some platforms. `LLM_CPU_THREADS` can optionally set the number of CPU threads when CPU execution is selected.

### Candidate naturalness scoring

`EmbedderLLM` preserves the fixed embedding positions and candidate-character rule. For valid candidates only, it combines the candidate's model log-probability with a local score based on readable token shape, punctuation, whitespace, repetition, topic context, and sentence continuity. The default setting is:

```python
EmbedderLLM.NATURALNESS_WEIGHT = 0.75
```

Increase this value carefully if readable continuations should have more influence. The Qwen probability remains the primary signal, and candidates that do not satisfy the hidden-character constraint are never selected. Selection details are written to the normal logger, including candidates checked, valid candidates, probability, naturalness score, and final score.

## Validation commands

Run the complete compile check:

```powershell
python -m py_compile app/config.py app/crypto/mapping.py app/crypto/position_generator.py app/llm/generator.py app/llm/embedder.py app/extraction/extractor.py app/evaluation/metrics.py app/evaluation/naturalness.py app/demo.py
```

Run the tests and official fixed-position demo:

```powershell
python -m pytest -q
python -m app.demo
```

The demo reports original and mapped secrets, deterministic positions, every embedding-position check, generated cover text, extraction results, naturalness validation, performance, and the final `END-TO-END TEST: PASS` or `FAIL` result. Final PASS requires both exact recovery and passing naturalness checks.

## Quick Qwen generation

For a short generation-only check:

```powershell
python -c "from app.llm.generator import LLMGenerator; g = LLMGenerator(); print(g.generate('Write a short sci-fi opening paragraph about a city under glass.', temperature=0.8, top_k=40, max_new_tokens=30, deterministic=True))"
```

## Evaluation and paper alignment

- [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) records exact matches, prototype assumptions, deviations, and future work.
- `app.evaluation.metrics` contains data-driven aggregate metric calculations.
- `app.evaluation.naturalness` reports lightweight repetition and sentence-shape statistics without heavyweight NLP dependencies.

## ECDHE transport encryption

Run the separate, in-process Alice/Bob integration demo with:

```powershell
python -m app.crypto.ecdhe_demo
```

Both peers generate fresh X25519 key pairs and simulate exchanging their
public keys and a shared 16-byte salt in the same Python process. They keep
their private keys secret and independently call `derive_session_keys()` with
the same salt and associated data. Each side receives a send key and a
different receive key. The demo encrypts and decrypts a message in both
directions using those keys and the existing `app.crypto.aead` AES-256-GCM
implementation. It prints each recovered message and a final `PASS`.

```python
from app.crypto.aead import decrypt, encrypt
from app.crypto.ecdhe import derive_session_keys, generate_key_pair

alice_private, alice_public = generate_key_pair()
bob_private, bob_public = generate_key_pair()
salt = b"shared-salt-1234"  # Alice shares this with her public key.
context = b"session context"

alice_send, alice_receive = derive_session_keys(alice_private, bob_public, salt, context)
bob_send, bob_receive = derive_session_keys(bob_private, alice_public, salt, context)

packet = encrypt(b"hidden payload", alice_send, context)
plaintext = decrypt(
	packet["ciphertext"], packet["tag"], packet["nonce"], bob_receive, context
)
```

The existing `encrypt_for_peer()` / `decrypt_from_peer()` helpers remain
available for one-way encryption to a recipient's long-term public key. They
use a fresh sender ephemeral key per message and are separate from this
bilateral simulation. X25519 and HKDF are provided by the `cryptography`
package. AES-GCM is provided by the project's existing `app.crypto.aead`
module; the ECDHE integration reuses it and does not implement another AES-GCM
layer. This is not a live network handshake, and public-key authentication is
not implemented. In production, unauthenticated public keys are vulnerable to
man-in-the-middle attacks.

## Key Modules

### LLM Generation

The generator module wraps Hugging Face Transformers and exposes a clean interface for prompt-based generation with configurable temperature, top-k, and max token count.

### Character Mapping

A reversible mapping module encodes and decodes character sequences for embedding and extraction.

### Position Generation

A deterministic SHAKE128-inspired position generator creates embedding positions from secret material.

### EmbedderLLM

This is the main Phase 1 embedding engine, responsible for placing the hidden characters into generated text while maintaining a natural story flow.

### Extraction Prototype

This recovers the embedded character sequence using the same positions and story content.

## Notes

- The project is intentionally kept modular so future paper-specific updates can be incorporated without rewriting the whole system.
- The current implementation follows the research requirement for a defensible, modular Phase 1 core.
- UI and deployment work are not included in this phase.

## Status

The Phase 1 implementation is complete, and the test suite is intended to pass once the Python dependencies are installed in a working interpreter environment.
