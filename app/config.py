"""Global configuration for the Phase 1 LLM steganography prototype.

This file centralizes model selection and safe default generation settings so the
implementation can change models later without modifying the generator logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the Hugging Face generation backend."""

    model_name: str = os.getenv(
        "LLM_MODEL_NAME",
        "Qwen/Qwen2.5-0.5B-Instruct",
    )
    max_new_tokens_default: int = 128
    temperature_default: float = 0.7
    top_k_default: int = 40
    device: str = "cpu"
    seed: int | None = None


DEFAULT_LLM_CONFIG = LLMConfig()


def get_model_name() -> str:
    """Return the currently configured model name."""

    return DEFAULT_LLM_CONFIG.model_name
