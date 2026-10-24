# LLM-SHIELD

This repository contains the Phase 1 implementation of an LLM-based covert communication and steganography system inspired by the research paper specification.

## Project Overview

The project aims to build a research-oriented prototype for covert communication using large language models. The current implementation focuses on the core Phase 1 components and includes a localhost-only PSK demonstration UI:

- LLM generation engine
- reversible character mapping
- cryptographic position generation
- EmbedderLLM prototype
- extraction prototype
- X25519 ECDHE key exchange, PBKDF2 DK1/DK2 derivation, and AES-256-GCM encryption
- end-to-end validation experiment

## Current Phase

Phase 1 is the core implementation stage and includes the essential research contribution for the project. The web demonstration is intended for local academic review; it is not a production deployment or a claim of undetectability. The existing two-terminal TCP workflow supports both PSK and ECDHE. The web interface currently integrates the real PSK pipeline; it does not simulate or replace the TCP ECDHE exchange.

## Local demonstration UI

Run the following commands in PowerShell from the repository directory:

```powershell
cd "C:\Users\MY PC\llm-shield\project"
.\.venv\Scripts\python.exe -m app.web_server
```

Keep that PowerShell window open while using the UI. In your browser's address
bar, open **http://127.0.0.1:8765** (do not search for the URL). The server
binds to localhost only. If the virtual environment has not been created, use
the setup steps below first. Install the project requirements with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The first complete sender run initializes or downloads the configured Hugging
Face model, so it may take several minutes and requires internet access if the
model is not already cached. Keep the browser tab open while the model runs.
The interface uses the existing AES-GCM, PBKDF2, h4 mapping, SHAKE-128
position-generation, `EmbedderLLM`, extraction, and authenticated decryption
implementations. It shows public salt/nonce/AAD and derived-key lengths but
does not return derived key bytes or PSKs to the browser. Passwords are not
logged or persisted in browser storage.

Demo flow:

1. Enter a short plaintext, a PSK, and a carrier topic; optionally adjust
   temperature and top-k.
2. Select **Run complete sender** and inspect the resulting cryptographic,
   mapping, position, embedding, and carrier outputs.
3. **Deliver carrier text to receiver** copies only the carrier into the
   editable receiver field. Enter the same PSK and run receiver extraction.
4. The receiver derives its own keys and positions. Plaintext appears only
   after AES-256-GCM authentication succeeds. Editing the carrier or using a
   different PSK demonstrates extraction/authentication failure.

The UI execution log records stage outcomes only. Its public-channel panel
contains carrier text; salt, nonce, and mapped length are separate local demo
metadata used by the receiver, not embedded into the carrier. ECDHE remains
available in the terminal TCP workflow documented below.

Sender intermediate outputs stream into the pipeline panel as each stage
finishes. Cryptography, payload, mapping, and target-position results appear
while the LLM embedding stage is still running. The embedding card reports
candidate generation, retries, and each character only after it passes
per-character verification; the carrier and verification outputs appear when
generation and sender-side checks complete.

### Step-by-step UI run

1. Open PowerShell and change to the project folder:
   ```powershell
   cd "C:\Users\MY PC\llm-shield\project"
   ```
2. If `.venv` does not exist yet, create it:
   ```powershell
   python -m venv .venv
   ```
3. Install dependencies (once, or after updating `requirements.txt`):
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
4. Start the UI server:
   ```powershell
   .\.venv\Scripts\python.exe -m app.web_server
   ```
   Leave this terminal running. If port `8765` is already in use, stop the
   other demo server or start this one on another port, for example:
   `.\.venv\Scripts\python.exe -m app.web_server --port 8766`.
5. Open **http://127.0.0.1:8765** in the browser address bar (use `8766` if
   you chose that port).
6. Enter a short message, a pre-shared password, and a carrier topic, then
   select **Run complete sender**. The first run may take several minutes
   while the configured model downloads or initializes. Intermediate stage
   outputs appear as the sender runs.
7. After the sender succeeds, select **Deliver carrier text to receiver**.
   Enter the same password in **Receiver PSK**, then select
   **Run receiver pipeline**. A successful run displays authenticated
   plaintext.
8. When finished, press **Ctrl+C** in the PowerShell window to stop the server.

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

### Stage 4: Run the TCP demonstration

Start the receiver in Terminal 1 and leave it listening:

```powershell
.\.venv\Scripts\python.exe server_b.py
```

Then run the sender in Terminal 2. The normal `app.demo` command is an
optional sender client and does not start a receiver:

```powershell
.\.venv\Scripts\python.exe -m app.demo --mode ecdhe --topic "A quiet city street at dusk" --secret "HI"
```

The equivalent direct sender command is
`.\.venv\Scripts\python.exe server_a.py --mode ecdhe --topic "A quiet city street at dusk" --secret "HI"`.
Use `--mode psk` instead for password-based key establishment; enter the same
password at both terminals when prompted. Password input is hidden and must
not be passed as a command-line argument.

The demo automatically selects an available LLM backend. Set `LLM_DEVICE=cpu`
to force CPU execution or `LLM_DEVICE=cuda` / `LLM_DEVICE=cuda:0` to request
CUDA. Longer messages require more generation and may take longer. The
embedding candidate scorer reuses its already-computed naturalness score
instead of rescoring the same candidate.

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

Run the tests and then follow the two-terminal TCP demonstration above:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The sender and receiver print the live TCP handshake, derivation, AEAD, mapping, embedding,
extraction, authenticated recovery, and end-to-end verification. The secret is
printed only in the sender's input section and by the receiver after successful
AES-GCM authentication; the PSK is never printed.

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

Both modes use the same TCP protocol between Server A and Server B. Start
Server B first in one PowerShell terminal:

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

Server B listens on `127.0.0.1:50505` by default and handles one connection
per process; restart it for another session. In PSK mode, enter the same
password at both terminals when prompted. To use the interactive mode picker,
run `.\.venv\Scripts\python.exe server_a.py`; it asks for the topic and secret
when they were not supplied as arguments. `app.demo` accepts the same sender
arguments but always connects to an already-running Server B.

In interactive ECDHE mode, Server A connects and completes the peer-confirmed
X25519 exchange before prompting for the secret. It sends the KDF salt and
derives DK1/DK2 only after the user enters that message. Supplying `--message`
or `--secret` bypasses the prompt, but the message is still not encrypted or
sent until key confirmation succeeds.

In ECDHE mode, the peers exchange their X25519 public keys as ten alternating
LLM-generated cover-text messages: each side splits its key into five
segments, maps each segment, embeds it with `EmbedderLLM`, and sends the
generated carrier for its peer to extract. Each side derives a pair of
position keys with PBKDF2-HMAC-SHA256, using the 16-byte random session ID
(sent as the PBKDF2 salt) and `LLM_SHIELD_PREVIOUS_SHARED_SECRET`. Server A
embeds its five segments with DK1 and extracts Server B's with DK2; Server B
embeds with DK2 and extracts Server A's with DK1. Position indices are
recomputed by the receiver and are not sent in the carrier frame. The setting
must have the same value at both servers. If unset, the empty string is used as
the PBKDF2 input so first-time bootstrap works; because that input and the salt
are public/default, these positions are not secret and this default does not
protect against an observer or an active man-in-the-middle. Configure a
previously shared secret on both servers if the key-carrier positions need to
be confidential. X25519 public-key authentication against an active
man-in-the-middle is not provided by this demo.

Set the optional bootstrap secret in both terminals when running separate
processes:

```powershell
$env:LLM_SHIELD_PREVIOUS_SHARED_SECRET = "your previously shared secret"
```

After all ten messages, both sides independently derive the X25519 shared
secret and confirm key possession. Only then does the sender request a secret
message (unless supplied with `--message` or `--secret`). In PSK mode, the
peers instead enter their shared password.

Both modes then share a PBKDF2 salt, derive 512 bits, and split them into DK1
and DK2. DK1 encrypts/authenticates with the existing AES-256-GCM
implementation. The mapped secret payload contains only the AEAD
authentication tag and ciphertext; for `HI` it is 36 h4 characters. DK2
drives SHAKE-128 position generation. Server A embeds the mapped secret
payload with `EmbedderLLM` and sends the cover text to Server B, which
regenerates positions, extracts, inverse maps, authenticates, and decrypts it.
Server A prints the live, character-by-character progress for the encrypted
message payload; Server B identifies the carrier as sender-generated and
prints the received text, extracted data, authentication result, and final
verification. Restart Server B before each new one-shot connection.

The mapped message length is computed from the actual UTF-8 plaintext size:
two h4 characters per encrypted byte, including the 16-byte GCM tag. For
example, the two-byte plaintext `HI` produces 36 mapped characters. A 28-
character frame can instead be one of the 7-byte X25519 public-key segments
used during the ECDHE handshake; the payload length is never hard-coded.

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




Run the server using these commands:

Terminal 1 — Server B:

.\.venv\Scripts\python.exe server_b.py

Terminal 2 — Server A:

.\.venv\Scripts\python.exe server_a.py --mode ecdhe --topic "A quiet city street at dusk" --secret HI