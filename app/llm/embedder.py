"""
LLM-SHIELD Phase 1
Paper-based fixed-position EmbedderLLM.

Core flow:

    TOPIC
      |
      v
    Qwen next-token candidates
      |
      v
    Move toward fixed position b_i
      |
      v
    At/near b_i:
        find candidate satisfying C_i == Story[b_i]
      |
      v
    Select the most probable natural candidate
      |
      v
    Continue to next target position

Important:
- Positions remain fixed.
- Secret characters are NEVER directly inserted.
- Embedded characters must come from LLM-generated tokens.
- Extraction can therefore read the same positions.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Sequence

from app.crypto.mapping import CharacterMap
from app.llm.generator import LLMGenerator

logger = logging.getLogger(__name__)


# ======================================================================
# RESULT
# ======================================================================

@dataclass
class EmbeddingResult:
    story: str
    embedded_characters: str
    positions: list[int]
    attempts: int


# ======================================================================
# EMBEDDER
# ======================================================================

class EmbedderLLM:

    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_K = 40

    MAX_TEMPERATURE = 0.9
    MAX_TOP_K = 60

    DEFAULT_RETRIES = 8

    # Candidate scoring.
    # Probability remains dominant because this is an LLM
    # candidate-selection algorithm.
    PROBABILITY_WEIGHT = 1.0
    NATURALNESS_WEIGHT = 0.35
    MIN_CANDIDATE_PROBABILITY = 1e-5

    # Don't allow generation to run forever.
    DEFAULT_MAX_STEPS = 300

    # Words which are safe to repeat.
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but",
        "is", "are", "was", "were", "to",
        "of", "in", "on", "at", "for",
        "with", "as", "by", "from",
        "he", "she", "they", "his", "her",
        "it", "this", "that", "has", "have",
        "had", "be", "been", "can", "will",
        "would", "could", "their", "there",
    }

    BAD_PATTERNS = [
        r"\bbookish\s+book\b",
        r"\bbook\s+book\b",
        r"\blibrary\s+library\b",
        r"\bboy\s+boy\b",
        r"\bgirl\s+girl\b",
        r"\btree\s+tree\b",
        r"\bstudent\s+student\b",
        r"\bquestion\b",
        r"\banswer\b",
        r"\bquiz\b",
        r"\bcalculate\b",
        r"\bsolve\b",
        r"\bequation\b",
        r"\bformula\b",
    ]

    def __init__(
        self,
        llm_generator: LLMGenerator | None = None,
        character_map: CharacterMap | None = None,
    ) -> None:

        self.llm_generator = (
            llm_generator
            if llm_generator is not None
            else LLMGenerator()
        )

        self.character_map = (
            character_map
            if character_map is not None
            else CharacterMap()
        )

    # ==================================================================
    # INPUT VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_inputs(
        topic: str,
        characters: str,
        positions: Sequence[int],
    ) -> None:

        if not isinstance(topic, str):
            raise TypeError("topic must be a string")

        if not topic.strip():
            raise ValueError("topic must not be empty")

        if not isinstance(characters, str):
            raise TypeError("characters must be a string")

        if not characters:
            raise ValueError("characters must not be empty")

        if positions is None:
            raise ValueError("positions must not be None")

        if len(characters) != len(positions):
            raise ValueError(
                "Number of characters must equal number of positions."
            )

        previous = -1

        for position in positions:

            if not isinstance(position, int):
                raise TypeError(
                    "Embedding positions must be integers."
                )

            if position < 0:
                raise ValueError(
                    "Embedding positions cannot be negative."
                )

            if position <= previous:
                raise ValueError(
                    "Embedding positions must be strictly increasing."
                )

            previous = position

    # ==================================================================
    # CHARACTER MATCH
    # ==================================================================

    @staticmethod
    def _character_matches(
        actual: str,
        required: str,
    ) -> bool:

        if not actual or not required:
            return False

        return actual.upper() == required.upper()

    @staticmethod
    def _model_prompt(topic: str, story: str) -> str:
        """Give Qwen narrative instructions while preserving cover indexing."""

        return (
            "Write a coherent, natural short story in ordinary English. "
            "Stay strongly related to this topic: "
            f"{topic.strip()}\n"
            "Avoid repetition, formulas, questions, technical language, "
            "and dictionary-like text. Do not mention hidden messages. "
            "Use complete sentences and maintain narrative continuity.\n"
            "Story so far:\n"
            f"{story}"
        )

    @staticmethod
    def verify_embedding(
        cover_text: str,
        mapped: str,
        positions: Sequence[int],
    ) -> bool:
        """Verify exact, collision-free fixed-position embedding."""

        if not isinstance(cover_text, str) or not isinstance(mapped, str):
            return False

        if len(mapped) != len(positions):
            return False

        if len(set(positions)) != len(positions):
            return False

        if any(not isinstance(position, int) or position < 0 for position in positions):
            return False

        if positions and max(positions) >= len(cover_text):
            return False

        return all(
            cover_text[position].upper() == expected.upper()
            for position, expected in zip(positions, mapped)
        )

    # ==================================================================
    # NATURALNESS OF ONE CANDIDATE
    # ==================================================================

    def _candidate_naturalness(
        self,
        story: str,
        token: str,
        topic: str,
    ) -> float:

        if not token:
            return 0.0

        text = story + token

        score = 0.65

        lower = text.lower()

        # --------------------------------------------------------------
        # Strongly penalize obvious repetition.
        # --------------------------------------------------------------

        for pattern in self.BAD_PATTERNS:

            if re.search(pattern, lower):
                score -= 0.65

        # --------------------------------------------------------------
        # Repeated word immediately before candidate.
        # --------------------------------------------------------------

        words = re.findall(r"[A-Za-z]+", lower)

        if len(words) >= 2:

            if words[-1] == words[-2]:
                if words[-1] not in self.STOPWORDS:
                    score -= 0.35

        # --------------------------------------------------------------
        # Candidate should not repeatedly introduce the same word.
        # --------------------------------------------------------------

        candidate_words = re.findall(
            r"[A-Za-z]+",
            token.lower(),
        )

        recent_words = words[-8:]

        for word in candidate_words:

            if (
                len(word) >= 4
                and word in recent_words
                and word not in self.STOPWORDS
            ):
                score -= 0.18

        # --------------------------------------------------------------
        # Topic relevance.
        # --------------------------------------------------------------

        topic_words = {
            word
            for word in re.findall(
                r"[A-Za-z]+",
                topic.lower(),
            )
            if len(word) >= 4
        }

        if topic_words.intersection(candidate_words):
            score += 0.2

        # --------------------------------------------------------------
        # Bad boundary.
        # --------------------------------------------------------------

        if re.search(r"[a-z][A-Z]", token):
            score -= 0.45

        # --------------------------------------------------------------
        # Technical-looking output.
        # --------------------------------------------------------------

        if re.search(
            r"(?:\\[A-Za-z]+|[$^_=]|[{}]|\d+\s*[=+*/-])",
            token,
        ):
            score -= 0.65

        # --------------------------------------------------------------
        # Natural punctuation.
        # --------------------------------------------------------------

        if token.strip() in {".", ",", "!", "?", ";", ":"}:
            score += 0.05

        return max(0.0, min(1.0, score))

    # ==================================================================
    # FINAL NATURALNESS VALIDATION
    # ==================================================================

    @staticmethod
    def _needs_completion(story: str) -> bool:

        stripped = story.strip()

        if not stripped:
            return True

        # A final word fragment is usually undesirable.
        last = stripped[-1]

        return last not in ".!?\"'”’"

    # ------------------------------------------------------------------

    @classmethod
    def _validate_cover_naturalness(
        cls,
        story: str,
        topic: str,
    ) -> dict[str, bool]:

        words = re.findall(
            r"[A-Za-z]+",
            story.lower(),
        )

        topic_words = {
            word
            for word in re.findall(
                r"[A-Za-z]+",
                topic.lower(),
            )
            if len(word) >= 4
        }

        # --------------------------------------------------------------
        # Immediate repetition
        # --------------------------------------------------------------

        repeated_words = bool(
            re.search(
                r"\b([A-Za-z]+)(?:\s+\1)+\b",
                story,
                re.IGNORECASE,
            )
        )

        # --------------------------------------------------------------
        # Repeated 3-grams
        # --------------------------------------------------------------

        ngrams = [
            tuple(words[i:i + 3])
            for i in range(
                max(0, len(words) - 2)
            )
        ]

        repeated_ngrams = (
            len(ngrams) != len(set(ngrams))
        )

        # --------------------------------------------------------------
        # Excessive nearby repetition.
        #
        # Do not punish common words.
        # --------------------------------------------------------------

        repeated_nearby = False

        for i in range(len(words)):

            for distance in range(
                2,
                min(5, len(words) - i),
            ):

                word = words[i]

                if (
                    word == words[i + distance]
                    and word not in cls.STOPWORDS
                    and len(word) >= 4
                ):
                    repeated_nearby = True
                    break

            if repeated_nearby:
                break

        # --------------------------------------------------------------
        # Technical/malformed text
        # --------------------------------------------------------------

        malformed_boundary = bool(
            re.search(
                r"[a-z][A-Z]",
                story,
            )
        )

        technical_pattern = bool(
            re.search(
                r"(?:\\[A-Za-z]+|[$^_=]|[{}]|"
                r"\d+\s*[=+*/-]|[<>])",
                story,
            )
        )

        # --------------------------------------------------------------
        # Topic relevance
        # --------------------------------------------------------------

        topic_relevance = bool(
            topic_words.intersection(words)
        )

        # --------------------------------------------------------------
        # Sentence completeness
        # --------------------------------------------------------------

        sentence_complete = not cls._needs_completion(story)

        return {
            "topic_relevance": topic_relevance,
            "repetition": (
                not repeated_words
                and not repeated_ngrams
                and not repeated_nearby
            ),
            "sentence_completeness": sentence_complete,
            "malformed_or_technical": (
                not malformed_boundary
                and not technical_pattern
            ),
        }

    # ==================================================================
    # SCORE CANDIDATE
    # ==================================================================

    def _score_candidate(
        self,
        story: str,
        token: str,
        probability: float,
        topic: str,
    ) -> float:

        probability = max(
            float(probability),
            1e-12,
        )

        log_probability = math.log(probability)
        probability_score = max(
            0.0,
            min(1.0, (log_probability + 12.0) / 12.0),
        )

        naturalness = self._candidate_naturalness(
            story=story,
            token=token,
            topic=topic,
        )

        return (
            self.PROBABILITY_WEIGHT * probability_score
            + self.NATURALNESS_WEIGHT * naturalness
        ) / (self.PROBABILITY_WEIGHT + self.NATURALNESS_WEIGHT)

    # ==================================================================
    # SELECT NORMAL TOKEN
    # ==================================================================

    def _select_normal_candidate(
        self,
        story: str,
        candidates,
        topic: str,
    ):

        if not candidates:
            return None

        best = None
        best_score = float("-inf")

        for candidate in candidates:

            token = candidate.token

            if not token:
                continue

            if candidate.probability < self.MIN_CANDIDATE_PROBABILITY:
                continue

            score = self._score_candidate(
                story=story,
                token=token,
                probability=candidate.probability,
                topic=topic,
            )

            if score > best_score:

                best_score = score
                best = candidate

        return best

    # ==================================================================
    # SELECT EMBEDDING CANDIDATE
    # ==================================================================

    def _select_embedding_candidate(
        self,
        story: str,
        candidates,
        character: str,
        position: int,
        topic: str,
    ):

        valid = []

        for candidate in candidates:

            token = candidate.token

            if not token:
                continue

            if candidate.probability < self.MIN_CANDIDATE_PROBABILITY:
                continue

            new_story = story + token

            # Candidate must actually reach the target position.
            if len(new_story) <= position:
                continue

            # Candidate must NOT skip the target incorrectly.
            if self._character_matches(
                new_story[position],
                character,
            ):

                naturalness = (
                    self._candidate_naturalness(
                        story=story,
                        token=token,
                        topic=topic,
                    )
                )

                score = self._score_candidate(
                    story=story,
                    token=token,
                    probability=candidate.probability,
                    topic=topic,
                )

                if naturalness < 0.25:
                    continue

                valid.append(
                    (
                        score,
                        candidate,
                        naturalness,
                    )
                )

        if not valid:
            return None

        # Highest combined probability/naturalness score.
        valid.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return valid[0]

    # ==================================================================
    # GENERATE UNTIL POSITION
    # ==================================================================

    def _generate_to_position(
        self,
        story: str,
        character: str,
        position: int,
        topic: str,
        temperature: float,
        top_k: int,
        max_steps: int,
        max_attempts: int,
        attempt_counter: list[int],
    ):

        for step in range(max_steps):

            if len(story) > position:
                if self._character_matches(
                    story[position],
                    character,
                ):
                    return story

                # We have crossed the position with the wrong character.
                return None

            candidates = (
                self.llm_generator
                .get_next_token_candidates(
                    prompt=self._model_prompt(topic, story),
                    top_k=top_k,
                    temperature=temperature,
                )
            )

            if not candidates:
                return None

            attempt_counter[0] += len(candidates)

            if attempt_counter[0] > max_attempts:
                raise RuntimeError(
                    "Maximum candidate evaluations exceeded."
                )

            # ----------------------------------------------------------
            # First check whether a candidate can directly satisfy
            # the target position.
            # ----------------------------------------------------------

            selected = self._select_embedding_candidate(
                story=story,
                candidates=candidates,
                character=character,
                position=position,
                topic=topic,
            )

            if selected is not None:

                score, candidate, naturalness = selected

                new_story = story + candidate.token

                logger.info(
                    "Selected embedding token=%r "
                    "target=%d probability=%.6f "
                    "naturalness=%.2f normalized_score=%.2f",
                    candidate.token,
                    position,
                    candidate.probability,
                    naturalness,
                    score,
                )

                return new_story

            # ----------------------------------------------------------
            # No direct candidate.
            #
            # Select a normal natural continuation, but only if it
            # does not cross the target.
            # ----------------------------------------------------------

            normal_candidates = []

            for candidate in candidates:

                token = candidate.token

                if not token:
                    continue

                new_story = story + token

                # Never cross the target with a wrong character.
                if len(new_story) > position:
                    continue

                normal_candidates.append(candidate)

            if not normal_candidates:
                return None

            normal = self._select_normal_candidate(
                story=story,
                candidates=normal_candidates,
                topic=topic,
            )

            if normal is None:
                return None

            story += normal.token

        return None

    # ==================================================================
    # EMBED ONE CHARACTER
    # ==================================================================

    def _embed_one_character(
        self,
        story: str,
        character: str,
        position: int,
        topic: str,
        temperature: float,
        top_k: int,
        max_retries: int,
        max_steps: int,
        max_attempts: int,
        attempt_counter: list[int],
    ):

        retry_temperature = temperature
        retry_top_k = top_k

        original_story = story

        for retry in range(1, max_retries + 1):

            logger.info(
                "Character %r target=%d retry=%d "
                "T=%.2f k=%d",
                character,
                position,
                retry,
                retry_temperature,
                retry_top_k,
            )

            # Every retry starts from the previous successful position.
            story = original_story

            result = self._generate_to_position(
                story=story,
                character=character,
                position=position,
                topic=topic,
                temperature=retry_temperature,
                top_k=retry_top_k,
                max_steps=max_steps,
                max_attempts=max_attempts,
                attempt_counter=attempt_counter,
            )

            if result is not None:

                if (
                    position < len(result)
                    and self._character_matches(
                        result[position],
                        character,
                    )
                ):
                    logger.info(
                        "Character %r successfully embedded "
                        "at position %d.",
                        character,
                        position,
                    )

                    return result

            # ----------------------------------------------------------
            # Paper-inspired retry adjustment.
            # ----------------------------------------------------------

            if retry_temperature < self.MAX_TEMPERATURE:

                retry_temperature = min(
                    self.MAX_TEMPERATURE,
                    round(
                        retry_temperature + 0.05,
                        2,
                    ),
                )

            elif retry_top_k < self.MAX_TOP_K:

                retry_top_k += 5
                retry_temperature = temperature

            else:

                # No more parameter expansion.
                break

        raise RuntimeError(
            f"Unable to embed character {character!r} "
            f"at position {position} after "
            f"{max_retries} retries."
        )

    # ==================================================================
    # MAIN EMBED
    # ==================================================================

    def embed(
        self,
        topic: str,
        characters: str,
        positions: Sequence[int],
        initial_story: str = "",
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        max_new_tokens: int = 128,
        max_attempts: int = 10000,
        max_retries: int = DEFAULT_RETRIES,
        deterministic: bool = False,
    ) -> EmbeddingResult:

        self._validate_inputs(
            topic=topic,
            characters=characters,
            positions=positions,
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

        if max_attempts <= 0:
            raise ValueError(
                "max_attempts must be > 0"
            )

        if max_retries <= 0:
            raise ValueError(
                "max_retries must be > 0"
            )

        self.llm_generator._load_backend()
        logger.info("LLM backend ready; reusing model for all candidates.")

        # --------------------------------------------------------------
        # Start story.
        # --------------------------------------------------------------

        if initial_story and initial_story.strip():
            story = initial_story.strip()
        else:
            story = topic.strip()

        # Do not allow topic itself to already pass a target position.
        for position in positions:

            if position < len(story):

                raise ValueError(
                    f"Initial story already exceeds target "
                    f"position {position}. "
                    "Use a shorter initial story."
                )

        attempt_counter = [0]

        logger.info("=" * 70)
        logger.info(
            "STARTING PAPER-BASED EmbedderLLM"
        )
        logger.info(
            "Model: %s",
            getattr(
                self.llm_generator,
                "model_name",
                "unknown",
            ),
        )
        logger.info(
            "Characters: %r",
            characters,
        )
        logger.info(
            "Positions: %s",
            list(positions),
        )
        logger.info(
            "Initial temperature: %.2f",
            temperature,
        )
        logger.info(
            "Initial top-k: %d",
            top_k,
        )

        # --------------------------------------------------------------
        # Embed every character sequentially.
        # --------------------------------------------------------------

        for index, (
            character,
            position,
        ) in enumerate(
            zip(
                characters,
                positions,
            )
        ):

            logger.info("-" * 70)
            logger.info(
                "Embedding %d/%d",
                index + 1,
                len(characters),
            )
            logger.info(
                "Required character: %r",
                character,
            )
            logger.info(
                "Target position: %d",
                position,
            )

            # The current story must never already be beyond target.
            if len(story) > position:

                raise RuntimeError(
                    f"Story already crossed target position "
                    f"{position}."
                )

            story = self._embed_one_character(
                story=story,
                character=character,
                position=position,
                topic=topic,
                temperature=temperature,
                top_k=top_k,
                max_retries=max_retries,
                max_steps=max_new_tokens * 4,
                max_attempts=max_attempts,
                attempt_counter=attempt_counter,
            )

            # Immediate verification.
            if (
                position >= len(story)
                or not self._character_matches(
                    story[position],
                    character,
                )
            ):

                raise RuntimeError(
                    f"Embedding validation failed at "
                    f"position {position}."
                )

        # --------------------------------------------------------------
        # Final validation.
        # --------------------------------------------------------------

        logger.info("=" * 70)
        logger.info(
            "FINAL EMBEDDING VALIDATION"
        )

        for character, position in zip(
            characters,
            positions,
        ):

            if position >= len(story):

                raise RuntimeError(
                    f"Position {position} is outside "
                    f"story length {len(story)}."
                )

            actual = story[position]

            passed = self._character_matches(
                actual,
                character,
            )

            logger.info(
                "position=%d expected=%r actual=%r PASS=%s",
                position,
                character,
                actual,
                passed,
            )

            if not passed:

                raise RuntimeError(
                    "Final embedding validation failed."
                )

        # --------------------------------------------------------------
        # Naturalness validation.
        # --------------------------------------------------------------

        naturalness = (
            self._validate_cover_naturalness(
                story=story,
                topic=topic,
            )
        )

        logger.info(
            "NATURALNESS VALIDATION: %s",
            naturalness,
        )

        logger.info(
            "Final story length: %d",
            len(story),
        )

        logger.info(
            "Total candidate evaluations: %d",
            attempt_counter[0],
        )

        # --------------------------------------------------------------
        # IMPORTANT:
        # Do not fail C == C' simply because the heuristic naturalness
        # checker dislikes the text. The primary Phase-1 correctness
        # condition is that the fixed-position characters are recovered.
        # Naturalness is reported separately.
        # --------------------------------------------------------------

        return EmbeddingResult(
            story=story,
            embedded_characters=characters,
            positions=list(positions),
            attempts=attempt_counter[0],
        )


__all__ = [
    "EmbeddingResult",
    "EmbedderLLM",
]