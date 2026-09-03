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
import time
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

    DEFAULT_RETRIES = 12

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
        self._embedding_stats = {
            "llm_calls": 0,
            "candidate_evaluations": 0,
            "retries": 0,
        }

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
        if actual is None or required is None:
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

    @staticmethod
    def _is_sentence_complete(story: str) -> bool:
        stripped = story.strip()
        if not stripped:
            return False
        last = stripped[-1]
        return last in ".!?\"'”’"

    @staticmethod
    def _is_repetitive_continuation(story: str, token: str) -> bool:
        if not token or not token.strip():
            return True

        text = (story + token).lower()
        words = re.findall(r"[A-Za-z]+", text)

        if len(words) >= 2 and words[-1] == words[-2]:
            return True

        if len(words) >= 3:
            recent = words[-8:]
            for i in range(len(recent) - 2):
                if recent[i] == recent[i + 2] and recent[i] != recent[i + 1]:
                    return True

        return False

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

    def _complete_cover_text(
        self,
        story: str,
        topic: str,
        max_tokens: int = 80,
    ) -> str:
        if not self._needs_completion(story):
            return story

        for temperature, top_k in (
            (0.75, 60),
            (0.85, 60),
            (0.90, 60),
            (0.80, 60),
        ):
            candidate_story = story

            for _ in range(max_tokens):
                candidates = self.llm_generator.get_next_token_candidates(
                    prompt=self._model_prompt(topic, candidate_story),
                    top_k=top_k,
                    temperature=temperature,
                )

                if not candidates:
                    break

                valid = []
                for candidate in candidates:
                    token = candidate.token
                    if not token:
                        continue

                    next_story = candidate_story + token
                    if self._is_repetitive_continuation(candidate_story, token):
                        continue

                    valid.append(candidate)

                if not valid:
                    break

                normal = self._select_normal_candidate(
                    story=candidate_story,
                    candidates=valid,
                    topic=topic,
                )

                if normal is None:
                    break

                candidate_story += normal.token

                if not self._needs_completion(candidate_story):
                    return candidate_story

            if not self._needs_completion(candidate_story):
                return candidate_story

        ended = story.rstrip()
        if ended and ended[-1] not in ".!?\"'”’":
            ended += "."
        return ended

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

            self._embedding_stats["candidate_evaluations"] += 1

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
        continuation_candidates=None,
    ):

        valid = []
        rejection_reasons = []

        for candidate in candidates:

            self._embedding_stats["candidate_evaluations"] += 1

            token = candidate.token

            if not token:
                rejection_reasons.append("empty token")
                continue

            new_story = story + token

            # Candidate must actually reach the target position.
            if len(new_story) <= position:
                if continuation_candidates is not None:
                    continuation_candidates.append(candidate)
                rejection_reasons.append("token does not reach target position")
                continue

            target_character = new_story[position]

            # Candidate must NOT skip the target incorrectly.
            if not self._character_matches(
                target_character,
                character,
            ):
                rejection_reasons.append(
                    f"target contains {target_character!r}, expected {character!r}"
                )
                continue

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
                rejection_reasons.append("naturalness score below 0.25")
                continue

            valid.append(
                (
                    score,
                    candidate,
                    naturalness,
                )
            )

        if not valid:
            self._last_failure_reason = (
                "no valid candidate"
                if not rejection_reasons
                else "no valid candidate: " + "; ".join(rejection_reasons[:3])
            )
            return None

        # Highest combined probability/naturalness score.
        valid.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return valid[0]

    # ==================================================================
    # SELECT SPACE CANDIDATE
    # ==================================================================

    def _select_space_candidate(
        self,
        story: str,
        candidates,
        position: int,
        topic: str,
        continuation_candidates=None,
    ):
        valid = []
        rejection_reasons = []

        for candidate in candidates:
            self._embedding_stats["candidate_evaluations"] += 1
            token = candidate.token
            if not token:
                rejection_reasons.append("empty token")
                continue

            new_story = story + token

            if len(new_story) <= position:
                if continuation_candidates is not None:
                    continuation_candidates.append(candidate)
                rejection_reasons.append("token does not reach target position")
                continue

            # SPACE validation is based on the resulting story character because
            # Qwen uses subword tokens with leading whitespace.
            target_character = new_story[position]
            if not (target_character == " " or target_character.isspace()):
                rejection_reasons.append(
                    f"target contains {target_character!r}, expected a space"
                )
                continue

            naturalness = self._candidate_naturalness(
                story=story,
                token=token,
                topic=topic,
            )

            # Do NOT apply MIN_CANDIDATE_PROBABILITY to SPACE.
            score = self._score_candidate(
                story=story,
                token=token,
                probability=candidate.probability,
                topic=topic,
            )

            valid.append((score, candidate, naturalness))

        if not valid:
            self._last_failure_reason = (
                "no valid space candidate"
                if not rejection_reasons
                else "no valid space candidate: " + "; ".join(rejection_reasons[:3])
            )
            return None

        valid.sort(key=lambda item: item[0], reverse=True)
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

        self._last_failure_reason = "generation did not reach the target position"

        for step in range(max_steps):

            if len(story) > position:
                if self._character_matches(
                    story[position],
                    character,
                ):
                    return story

                # We have crossed the position with the wrong character.
                self._last_failure_reason = (
                    f"target contains {story[position]!r}, expected {character!r}"
                )
                return None

            self._embedding_stats["llm_calls"] += 1
            candidates = (
                self.llm_generator
                .get_next_token_candidates(
                    prompt=self._model_prompt(topic, story),
                    top_k=top_k,
                    temperature=temperature,
                )
            )

            if not candidates:
                self._last_failure_reason = "Qwen returned no candidates"
                return None

            attempt_counter[0] += len(candidates)

            if attempt_counter[0] > max_attempts:
                self._last_failure_reason = (
                    f"maximum candidate attempts exceeded ({max_attempts})"
                )
                return None

            # ----------------------------------------------------------
            # First check whether a candidate can directly satisfy
            # the target position.
            # ----------------------------------------------------------

            normal_candidates = []
            if character == " ":
                selected = self._select_space_candidate(
                    story=story,
                    candidates=candidates,
                    position=position,
                    topic=topic,
                    continuation_candidates=normal_candidates,
                )
            else:
                selected = self._select_embedding_candidate(
                    story=story,
                    candidates=candidates,
                    character=character,
                    position=position,
                    topic=topic,
                    continuation_candidates=normal_candidates,
                )

            if selected is not None:

                _, candidate, _ = selected
                return story + candidate.token

            # ----------------------------------------------------------
            # No direct candidate.
            #
            # Select a normal natural continuation, but only if it
            # does not cross the target.
            # ----------------------------------------------------------

            if not normal_candidates:
                if not getattr(self, "_last_failure_reason", ""):
                    self._last_failure_reason = "all candidates would cross the target"
                return None

            normal = self._select_normal_candidate(
                story=story,
                candidates=normal_candidates,
                topic=topic,
            )

            if normal is None:
                self._last_failure_reason = "no natural continuation candidate"
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
        """
        Embed one required character at the fixed target position.

        Use the fixed retry order required by the paper workflow.
        """

        original_story = story
        last_failure_reason = "unknown embedding failure"
        character_started = time.perf_counter()
        character_stats_start = dict(self._embedding_stats)
        retry_schedule = [
            (0.70, 40), (0.70, 50), (0.70, 60),
            (0.75, 40), (0.75, 50), (0.75, 60),
            (0.80, 40), (0.80, 50), (0.80, 60),
        ]

        for retry_round in range(1, max_retries + 1):
            if retry_round > 1:
                # A fresh candidate query is the explicit state change that
                # justifies another pass through the paper schedule.
                candidate_cache = getattr(self.llm_generator, "_candidate_cache", None)
                if candidate_cache is None:
                    break
                candidate_cache.clear()

            for schedule_number, (retry_temperature, retry_top_k) in enumerate(
                retry_schedule,
                start=1,
            ):
                retry_number = ((retry_round - 1) * len(retry_schedule)) + schedule_number
                self._embedding_stats["retries"] += 1
                print(
                    f"Retry {retry_number}: T={retry_temperature:.2f} "
                    f"k={retry_top_k}"
                )

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
                        print("Status: Embedded successfully")
                        elapsed = time.perf_counter() - character_started
                        print(
                            "Character stats: "
                            f"LLM calls={self._embedding_stats['llm_calls'] - character_stats_start['llm_calls']}, "
                            f"candidate evaluations={self._embedding_stats['candidate_evaluations'] - character_stats_start['candidate_evaluations']}, "
                            f"retries={self._embedding_stats['retries'] - character_stats_start['retries']}, "
                            f"time={elapsed:.2f}s"
                        )
                        return result

                last_failure_reason = getattr(
                    self,
                    "_last_failure_reason",
                    "candidate rejected or generation did not reach the position",
                )
                print(f"Candidate rejected: {last_failure_reason}")
                print("Retrying same character and position...")

        elapsed = time.perf_counter() - character_started
        print(
            "Character stats: "
            f"LLM calls={self._embedding_stats['llm_calls'] - character_stats_start['llm_calls']}, "
            f"candidate evaluations={self._embedding_stats['candidate_evaluations'] - character_stats_start['candidate_evaluations']}, "
            f"retries={self._embedding_stats['retries'] - character_stats_start['retries']}, "
            f"time={elapsed:.2f}s"
        )
        raise RuntimeError(
            f"Unable to embed character '{character}' at position {position} "
            f"after {max_retries} retry rounds and all retry configurations. "
            f"Last rejection reason: {last_failure_reason}."
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
        self._embedding_stats = {
            "llm_calls": 0,
            "candidate_evaluations": 0,
            "retries": 0,
        }

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

            display_char = "SPACE" if character == " " else character
            print("--------------------------------------------------")
            print(f"Embedding {index + 1}/{len(characters)}")
            print(f"Character: {display_char}")
            print(f"Position: {position}")

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

            if not passed:

                raise RuntimeError(
                    "Final embedding validation failed."
                )

        # --------------------------------------------------------------
        # Extend only the unfinished tail after the last embedded position.
        # The fixed-position payload itself remains unchanged.
        # --------------------------------------------------------------

        story = self._complete_cover_text(
            story=story,
            topic=topic,
            max_tokens=80,
        )

        naturalness = self._validate_cover_naturalness(
            story=story,
            topic=topic,
        )

        logger.info(
            "Final cover naturalness: %s",
            naturalness,
        )

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