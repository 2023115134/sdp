# LLM-SHIELD

This repository contains the Phase 1 implementation of an LLM-based covert communication and steganography system inspired by the research paper specification.

## Project Overview

The project aims to build a research-oriented prototype for covert communication using large language models. The current implementation focuses on the core Phase 1 components:

- LLM generation engine
- reversible character mapping
- cryptographic position generation
- EmbedderLLM prototype
- extraction prototype
- X25519 ECDHE key exchange, PBKDF2 DK1/DK2 derivation, and AES-256-GCM encryption
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

### 3. Use the project interpreter

Use `.venv\Scripts\python.exe` explicitly in every terminal. This avoids
accidentally running the Windows Store/global Python and importing packages
from its user site-packages:

### 4. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The default model is `Qwen/Qwen2.5-0.5B-Instruct`. Hugging Face may download it the first time a model-backed command runs. Keep the terminal connected to the internet for that first run.

Story generation is unseeded by default, so repeated runs with the same topic may produce different Qwen cover text. Mapping and SHAKE128 position generation remain deterministic. Pass an explicit `seed` to `LLMGenerator` when reproducible model output is needed; `deterministic=True` is also available for the generator test path.

## Run the tests

Run the complete automated test suite with the project interpreter:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected result:

```text
16 passed
```

## Run the paper-based workflow

Run these commands in this order using the project interpreter.

### Stage 1: Test the Qwen generator

```powershell
.\.venv\Scripts\python.exe -m app.llm.generator
```

This loads Qwen, generates a short continuation, prints top-k candidates, and verifies that their probabilities are sorted.

### Stage 2: Test one-character embedding

```powershell
.\.venv\Scripts\python.exe -m app.llm.embedder
```

This uses one hidden character, `E`, at position `50`. It has bounded retries and candidate evaluations for laptop safety. Candidate selection keeps only tokens that satisfy the exact fixed embedding position, then ranks them by model log-probability with local naturalness and context signals.

### Stage 3: Test extraction

```powershell
.\.venv\Scripts\python.exe -m app.extraction.extractor
```

This independently extracts and decodes the known `HELLO` payload.

### Stage 4: Run the complete demo

```powershell
.\.venv\Scripts\python.exe -m app.demo
```

The demo first prompts for PSK or ECDHE, followed by topic and secret message.
It connects to Server B for either mode. Run Server B in another terminal
with `.\.venv\Scripts\python.exe server_b.py` before starting the demo.

The demo performs:

```text
secret message
	-> h4 character mapping
	-> PBKDF2-derived DK1/DK2
	-> AES-256-GCM encryption
	-> h4 character mapping
	-> SHAKE-128 position generation
	-> fixed target positions
	-> Qwen candidate-token embedding
	-> TCP cover-text transmission
	-> position-based extraction
	-> inverse mapping
	-> AES-GCM authentication and decryption
	-> recovered message
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
.\.venv\Scripts\python.exe -m py_compile app/config.py app/crypto/mapping.py app/crypto/position_generator.py app/llm/generator.py app/llm/embedder.py app/extraction/extractor.py app/evaluation/metrics.py app/evaluation/naturalness.py app/demo.py
```

Run the tests and official fixed-position demo:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m app.demo
```

The demo reports original and mapped secrets, deterministic positions, every embedding-position check, generated cover text, extraction results, naturalness validation, performance, and the final `END-TO-END TEST: PASS` or `FAIL` result. Final PASS requires both exact recovery and passing naturalness checks.

## Quick Qwen generation

For a short generation-only check:

```powershell
.\.venv\Scripts\python.exe -c "from app.llm.generator import LLMGenerator; g = LLMGenerator(); print(g.generate('Write a short sci-fi opening paragraph about a city under glass.', temperature=0.8, top_k=40, max_new_tokens=30, deterministic=True))"
```

## Evaluation and paper alignment

- [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) records exact matches, prototype assumptions, deviations, and future work.
- `app.evaluation.metrics` contains data-driven aggregate metric calculations.
- `app.evaluation.naturalness` reports lightweight repetition and sentence-shape statistics without heavyweight NLP dependencies.

## PSK and ECDHE TCP communication

Both modes use the same detailed LLM-SHIELD stages and run between separate
Server A and Server B processes. From the project folder, start the receiver
first in one PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe server_b.py
```

For ECDHE, connect the sender in the other terminal:

```powershell
.\.venv\Scripts\python.exe server_a.py --mode ecdhe --message HI --topic "A quiet city street at dusk"
```

For PSK, use the same command with `--mode psk`:

```powershell
.\.venv\Scripts\python.exe server_a.py --mode psk --message HI --topic "A quiet city street at dusk"
```

Server B listens on `127.0.0.1:50505` by default. In PSK mode, enter the same
password at both terminals when prompted. To use the interactive mode picker,
run `.\.venv\Scripts\python.exe server_a.py` or
`.\.venv\Scripts\python.exe -m app.demo`; choose mode first and then enter the
topic and message.

In ECDHE mode, Server A and Server B exchange their X25519 public keys over
TCP and independently derive and confirm the raw shared secret. The keys are
separate handshake messages and are not embedded in the secret-message
payload. Both modes then share a PBKDF2 salt, derive 512 bits, and split them
into DK1 and DK2. DK1 encrypts/authenticates with the existing AES-256-GCM
implementation. The mapped payload contains only the AEAD authentication tag
and ciphertext; for `HI` it is 36 h4 characters. DK2 drives SHAKE-128 position
generation. Server A embeds the mapped payload with `EmbedderLLM` and sends
the cover text to Server B, which regenerates positions, extracts, inverse
maps, authenticates, and decrypts it. Both terminals display their respective
pipeline stages. Restart Server B before each new one-shot connection.

The in-process, bidirectional HKDF example remains available in
`python -m app.crypto.ecdhe_demo`:

```python
from app.crypto.aead import decrypt, encrypt
from app.crypto.ecdhe import derive_session_keys, generate_key_pair

alice_private, alice_public = generate_key_pair()
bob_private, bob_public = generate_key_pair()
salt = b"shared-salt-1234"
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
use a fresh sender ephemeral key per message and remain unchanged. The TCP
example uses the raw X25519 shared secret with the project's PBKDF2 derivation
for its covert-message pipeline. The demo does not authenticate peer
identities: unauthenticated public keys are vulnerable to man-in-the-middle
attacks.

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
