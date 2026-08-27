"""Dependency-free cover-text quality summaries."""

from __future__ import annotations

import re
from collections import Counter


def summarize_cover_text(text: str) -> dict[str, float | int]:
    """Return observable repetition and sentence-shape statistics."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    tokens = re.findall(r"\b[\w']+\b", text.lower())
    sentences = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    repeated_tokens = sum(count - 1 for count in Counter(tokens).values() if count > 1)
    repeated_phrases = sum(
        count - 1
        for count in Counter(
            " ".join(tokens[index:index + 3])
            for index in range(max(0, len(tokens) - 2))
        ).values()
        if count > 1
    )
    sentence_lengths = [len(re.findall(r"\b[\w']+\b", sentence)) for sentence in sentences]
    malformed_fragments = sum(
        not sentence[-1].isalnum()
        for sentence in sentences
        if sentence
    )

    return {
        "text_length": len(text),
        "token_count": len(tokens),
        "repeated_token_count": repeated_tokens,
        "repeated_three_word_phrase_count": repeated_phrases,
        "sentence_count": len(sentences),
        "average_sentence_length": (
            sum(sentence_lengths) / len(sentence_lengths)
            if sentence_lengths else 0.0
        ),
        "obvious_malformed_fragments": malformed_fragments,
    }
