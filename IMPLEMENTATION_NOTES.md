# Implementation Notes

## Phase 1 scope

This project is implementing the research paper's LLM covert communication / steganography approach in staged phases. The current milestone is the core LLM generation engine.

## Paper algorithm

The research paper defines an LLM-based steganography scheme that uses generation parameters such as temperature and top-k, plus cryptographic embedding positions for secret placement. This repository follows that concept at a high architectural level without pretending to reproduce every detail before the remaining phases are implemented.

## Our implementation

- A reusable `LLMGenerator` wraps Hugging Face Transformers.
- Configuration is centralized in `app/config.py`.
- The generator exposes a clean interface: `generate(prompt, temperature, top_k, max_new_tokens)`.
- Deterministic generation is supported with explicit seeds for testing.

## Assumptions

- The initial model is `Qwen/Qwen2.5-0.5B-Instruct`; set `LLM_MODEL_NAME` to use a different compatible model.
- Model selection is configurable and can be changed later without modifying generation logic.
- The embedder selects decoded candidate tokens at exact character positions. It is paper-inspired and documented as a prototype rather than an exact reproduction.

## Deviations / guardrails

- No UI or authentication layers are included in Phase 1.
- No custom cryptographic primitives are added yet.
- This implementation stays deliberately modular so the later paper-specific algorithms can be inserted without rewriting the entire project.
