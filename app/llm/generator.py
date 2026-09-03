"""
LLM Generator for LLM-Shield Phase 1.

This module provides:
1. Hugging Face causal language model loading
2. Text generation
3. Next-token probability extraction
4. Top-k candidate token generation
5. CPU execution suitable for the current laptop

The embedding module uses get_next_token_candidates()
to select tokens according to their probabilities.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Optional

import torch

from app.config import DEFAULT_LLM_CONFIG

logger = logging.getLogger(__name__)


# ============================================================
# TOKEN CANDIDATE
# ============================================================

@dataclass
class TokenCandidate:
    """
    Represents one possible next token predicted by the LLM.
    """

    token_id: int
    token: str
    probability: float


# ============================================================
# LLM GENERATOR
# ============================================================

class LLMGenerator:
    """
    Wrapper around a Hugging Face causal language model.

    The default model is configured in app.config and can be replaced
    with another compatible Hugging Face causal language model.

    The implementation is intentionally modular so that
    the model can later be replaced by another causal LLM.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:

        self.model_name = (
            model_name
            or DEFAULT_LLM_CONFIG.model_name
        )

        self.device = (
            device
            or DEFAULT_LLM_CONFIG.device
        )

        self.seed = (
            seed
            if seed is not None
            else DEFAULT_LLM_CONFIG.seed
        )

        self._tokenizer = None
        self._model = None
        self._logit_cache: dict[str, torch.Tensor] = {}
        self._candidate_cache: dict[tuple[str, int, float], list[TokenCandidate]] = {}

        logger.info(
            "Initializing LLM generator "
            "with model=%s device=%s seed=%s",
            self.model_name,
            self.device,
            self.seed,
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def _load_backend(self) -> None:
        """
        Load tokenizer and model lazily.
        """

        if (
            self._tokenizer is not None
            and self._model is not None
        ):
            return

        try:

            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
            )

        except ImportError as exc:

            raise RuntimeError(
                "Transformers is not installed."
            ) from exc

        # Avoid Windows CPU-threading crashes while materializing the
        # model weights in the research prototype. This keeps the
        # algorithm, configuration, and model selection unchanged.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        logger.info(
            "Loading tokenizer: %s",
            self.model_name,
        )

        try:

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )

            logger.info(
                "Loading model: %s",
                self.model_name,
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
            )

        except Exception as exc:

            logger.exception(
                "Failed to load model."
            )

            raise RuntimeError(
                f"Unable to initialize model '{self.model_name}'. "
                "Set LLM_MODEL_NAME to a cached local model or "
                "enable internet access so the model can be downloaded."
            ) from exc

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        if self.device == "cuda":

            if not torch.cuda.is_available():

                logger.warning(
                    "CUDA requested but unavailable. "
                    "Falling back to CPU."
                )

                self.device = "cpu"

            else:

                model.to("cuda")

        else:

            model.to("cpu")

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        if tokenizer.pad_token_id is None:

            tokenizer.pad_token = (
                tokenizer.eos_token
            )

        model.eval()

        self._tokenizer = tokenizer
        self._model = model

        logger.info(
            "Loaded model %s successfully",
            self.model_name,
        )

    # ========================================================
    # GENERATE TEXT
    # ========================================================

    def generate(
        self,
        prompt: str,
        temperature: float = DEFAULT_LLM_CONFIG.temperature_default,
        top_k: int = DEFAULT_LLM_CONFIG.top_k_default,
        max_new_tokens: int = DEFAULT_LLM_CONFIG.max_new_tokens_default,
        seed: Optional[int] = None,
        deterministic: bool = False,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 0,
    ) -> str:
        """
        Generate natural language continuation.
        """

        if not prompt or not prompt.strip():

            raise ValueError(
                "Prompt cannot be empty."
            )

        if temperature <= 0:

            raise ValueError(
                "temperature must be > 0"
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be > 0"
            )

        if max_new_tokens <= 0:

            raise ValueError(
                "max_new_tokens must be > 0"
            )

        if repetition_penalty < 1.0:
            raise ValueError(
                "repetition_penalty must be >= 1.0"
            )

        if no_repeat_ngram_size < 0:
            raise ValueError(
                "no_repeat_ngram_size must be >= 0"
            )

        self._load_backend()

        tokenizer = self._tokenizer
        model = self._model

        if tokenizer is None or model is None:

            raise RuntimeError(
                "LLM backend was not initialized."
            )

        # ----------------------------------------------------
        # Seed
        # ----------------------------------------------------

        generation_seed = (
            seed
            if seed is not None
            else self.seed
        )

        if generation_seed is not None:

            random.seed(generation_seed)
            torch.manual_seed(generation_seed)

            if torch.cuda.is_available():

                torch.cuda.manual_seed_all(
                    generation_seed
                )

        # ----------------------------------------------------
        # Tokenize
        # ----------------------------------------------------

        encoded = tokenizer(
            prompt,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        with torch.no_grad():

            output = model.generate(

                **encoded,

                do_sample=not deterministic,

                temperature=(
                    temperature
                    if not deterministic
                    else 1.0
                ),

                top_k=top_k,

                max_new_tokens=max_new_tokens,

                repetition_penalty=repetition_penalty,

                no_repeat_ngram_size=no_repeat_ngram_size,

                pad_token_id=(
                    tokenizer.pad_token_id
                ),

                eos_token_id=(
                    tokenizer.eos_token_id
                ),
            )

        # ----------------------------------------------------
        # Remove original prompt
        # ----------------------------------------------------

        prompt_length = (
            encoded["input_ids"].shape[1]
        )

        generated_tokens = output[
            0,
            prompt_length:
        ]

        generated_text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return generated_text.strip()

    # ========================================================
    # NEXT TOKEN CANDIDATES
    # ========================================================

    def get_next_token_candidates(
        self,
        prompt: str,
        top_k: int = 40,
        temperature: float = 1.0,
    ) -> list[TokenCandidate]:
        """
        Return the top-k possible next tokens predicted by the model.

        Each candidate contains:

            token_id
            token text
            probability

        Probabilities are sorted from highest to lowest.
        """

        if not prompt or not prompt.strip():

            raise ValueError(
                "Prompt cannot be empty."
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be > 0"
            )

        if temperature <= 0:

            raise ValueError(
                "temperature must be > 0"
            )

        cache_key = (prompt, int(top_k), round(float(temperature), 3))
        cached = self._candidate_cache.get(cache_key)
        if cached is not None:
            return cached

        self._load_backend()

        tokenizer = self._tokenizer
        model = self._model

        if tokenizer is None or model is None:

            raise RuntimeError(
                "LLM backend was not initialized."
            )

        # ----------------------------------------------------
        # Tokenize prompt
        # ----------------------------------------------------

        encoded = tokenizer(
            prompt,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        # ----------------------------------------------------
        # Reuse the same prompt logits across retries and top-k
        # variations. The underlying story context has not changed,
        # so there is no need to run the model forward pass again
        # merely to rescale or re-rank the same vocabulary.
        # ----------------------------------------------------

        prompt_key = prompt
        logits = self._logit_cache.get(prompt_key)

        if logits is None:
            with torch.no_grad():
                outputs = model(
                    **encoded
                )
                logits = outputs.logits[:, -1, :].detach().cpu()

            if len(self._logit_cache) >= 128:
                self._logit_cache.pop(next(iter(self._logit_cache)))

            self._logit_cache[prompt_key] = logits
        else:
            logits = logits.to(self.device)

        # ----------------------------------------------------
        # Temperature scaling
        # ----------------------------------------------------

        logits = logits / temperature

        # ----------------------------------------------------
        # Convert logits -> probabilities
        # ----------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        # ----------------------------------------------------
        # Top-k
        # ----------------------------------------------------

        actual_k = min(
            top_k,
            probabilities.shape[-1]
        )

        top_probabilities, top_ids = torch.topk(
            probabilities,
            k=actual_k,
            dim=-1,
        )

        # ----------------------------------------------------
        # Build candidates
        # ----------------------------------------------------

        candidates: list[TokenCandidate] = []

        for probability, token_id in zip(
            top_probabilities[0],
            top_ids[0],
        ):

            token_id_int = int(
                token_id.item()
            )

            token_text = tokenizer.decode(
                [token_id_int]
            )

            probability_float = float(
                probability.item()
            )

            candidates.append(
                TokenCandidate(
                    token_id=token_id_int,
                    token=token_text,
                    probability=probability_float,
                )
            )

        if len(self._candidate_cache) >= 512:
            self._candidate_cache.pop(next(iter(self._candidate_cache)))
        self._candidate_cache[cache_key] = candidates

        return candidates


# ============================================================
# TEST
# ============================================================

def main() -> None:

    print("=" * 70)
    print("LLM GENERATOR TEST")
    print("=" * 70)

    prompt = (
        "A student is walking through "
        "a beautiful city"
    )

    print("\nPrompt:")
    print(prompt)

    # --------------------------------------------------------
    # Initialize
    # --------------------------------------------------------

    generator = LLMGenerator()

    # --------------------------------------------------------
    # Test generation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST 1: TEXT GENERATION")
    print("=" * 70)

    generated = generator.generate(
        prompt=prompt,
        temperature=0.8,
        top_k=40,
        max_new_tokens=30,
        deterministic=False,
    )

    print("\nGenerated text:")
    print("-" * 70)
    print(generated)

    # --------------------------------------------------------
    # Test candidates
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST 2: TOP-40 NEXT-TOKEN CANDIDATES")
    print("=" * 70)

    candidates = (
        generator.get_next_token_candidates(
            prompt=prompt,
            top_k=40,
            temperature=1.0,
        )
    )

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"{index:2d}. "
            f"ID={candidate.token_id:5d} | "
            f"Token={candidate.token!r:15s} | "
            f"Probability={candidate.probability:.6f}"
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(
        f"Number of candidates: "
        f"{len(candidates)}"
    )

    if len(candidates) == 40:

        print("Top-k candidate test: PASS")

    else:

        print("Top-k candidate test: FAIL")

    probabilities = [
        candidate.probability
        for candidate in candidates
    ]

    sorted_correctly = all(
        probabilities[i]
        >= probabilities[i + 1]
        for i in range(
            len(probabilities) - 1
        )
    )

    print(
        "Probabilities sorted descending:",
        sorted_correctly,
    )

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()


__all__ = [
    "LLMGenerator",
    "TokenCandidate",
]