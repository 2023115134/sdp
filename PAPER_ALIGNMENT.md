# Paper Alignment

## Paper mechanism

The project is based on the paper's use of a language model's candidate tokens, a mapped character sequence, and key-derived increasing positions. A candidate is accepted only when its decoded text places the required character at the target character offset.

## Current implementation

- `CharacterMap` implements a reversible h4-style byte mapping.
- `PositionGenerator` derives deterministic positions from SHAKE128 and a 5-bit PRF chunk.
- `LLMGenerator` obtains top-k next-token candidates from the configured Qwen causal model.
- `EmbedderLLM` selects only returned model candidates and validates exact decoded Python string positions. It uses bounded search, caching, retries, and rollback boundaries.
- `Extractor` regenerates positions, reads mapped characters, and decodes them.

## Alignment status

**PARTIAL / PROTOTYPE IMPLEMENTATION.** The paper leaves some engineering details unspecified, including the exact PRF encoding, candidate ranking details, retry schedule, and natural-language quality criteria. This repository makes those assumptions explicit rather than claiming exact reproduction.

## Assumptions and deviations

- SHAKE128 is used as the project PRF with a domain separator and a 5-bit chunk; this is an implementation choice.
- Positions are character offsets in the decoded cover string, not tokenizer indices.
- Case-insensitive position validation is used because generated letters may be lowercase.
- Qwen/Qwen2.5-0.5B-Instruct is the configured local development model.
- No encryption, authentication, key exchange, PBKDF2, AEAD, or deployment layer is implemented in Phase 1.
- Wrong-key extraction is reported as key-dependent extraction validation, not as a cryptographic security proof.

## Future work

Implement and benchmark any paper-specific token-selection rule that is specified more precisely, add a separately designed encryption layer if required by the final architecture, and evaluate naturalness with independently justified metrics.
