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
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from app.crypto.mapping import CharacterMap
from app.llm.generator import LLMGenerator

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    story: str
    embedded_characters: str
    positions: list[int]
    attempts: int


class EmbedderLLM:

    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_K = 40

    MAX_TEMPERATURE = 0.9
    MAX_TOP_K = 100
    DIRECT_FALLBACK_TOP_K = 1000
    SPACE_FALLBACK_TOP_K = 1000
    DIRECT_FALLBACK_WINDOW = 96

    DEFAULT_RETRIES = 12

    # Candidate scoring.
    # Probability remains dominant because this is an LLM
    # candidate-selection algorithm.
    PROBABILITY_WEIGHT = 1.0
    NATURALNESS_WEIGHT = 0.45
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
            "Avoid unnecessary repetition of sentences, phrases, or content "
            "words from the story so far. Introduce a new event, action, or "
            "detail when it fits naturally. Vary the subject and verb where "
            "possible, but keep the narrative coherent. Avoid formulas, "
            "questions, technical language, Markdown, numbered lists, headings, "
            "and dictionary-like text. Do not mention hidden messages. Use "
            "complete sentences and maintain narrative continuity.\n"
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

        # Penalize repetition across the complete generated prefix, not only
        # the most recent tokens. Otherwise long fixed-position runs can drift
        # into a loop while each local window still appears acceptable.
        content_words = [
            word
            for word in words
            if len(word) >= 4 and word not in self.STOPWORDS
        ]
        if content_words:
            counts = Counter(content_words)
            dominant_count = max(counts.values())
            if dominant_count >= 4:
                score -= min(0.58, 0.10 * (dominant_count - 3))

        if len(words) >= 6:
            phrases = [
                tuple(words[index:index + 3])
                for index in range(len(words) - 2)
            ]
            repeated_phrase_count = sum(
                count - 1
                for count in Counter(phrases).values()
                if count > 1
            )
            score -= min(0.58, 0.07 * repeated_phrase_count)

        # Repeated names and named entities make a cover read like a broken
        # summary even when the individual words are otherwise common.
        proper_words = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
        proper_counts = Counter(proper_words)
        repeated_proper_words = sum(
            count - 2
            for count in proper_counts.values()
            if count > 2
        )
        score -= min(0.30, 0.08 * repeated_proper_words)

        if len(words) >= 2:

            if words[-1] == words[-2]:
                if words[-1] not in self.STOPWORDS:
                    score -= 0.35

        candidate_words = re.findall(
            r"[A-Za-z]+",
            token.lower(),
        )

        # --------------------------------------------------------------
        # Soft discourse-coherence checks.
        # --------------------------------------------------------------

        # Repeating a content word is sometimes natural, but repeating it
        # immediately or near the end of the prefix is usually a weak choice.
        # Keep this as a ranking penalty so fixed-position candidates remain
        # available when no cleaner token can carry the required character.
        if candidate_words:
            previous_words = words[:-len(candidate_words)]
            recent_content_words = {
                word
                for word in previous_words[-18:]
                if len(word) >= 4 and word not in self.STOPWORDS
            }
            repeated_candidate_words = sum(
                1
                for word in candidate_words
                if len(word) >= 4
                and word not in self.STOPWORDS
                and word in recent_content_words
            )
            score -= min(0.18, 0.06 * repeated_candidate_words)

        # Avoid starting several consecutive sentences with the same content
        # word. This catches flat generated prose without banning pronouns or
        # ordinary grammatical words.
        sentence_starts = re.findall(
            r"(?:^|[.!?]\s+)([A-Za-z]+)",
            story,
        )
        if candidate_words and sentence_starts:
            first_candidate_word = candidate_words[0]
            if (
                len(first_candidate_word) >= 4
                and first_candidate_word not in self.STOPWORDS
                and sentence_starts[-1].lower() == first_candidate_word
            ):
                score -= 0.12

        # Penalize punctuation collisions and sentence fragments lightly. A
        # token can still win when it is the only token satisfying a target.
        if story and token:
            previous_char = story[-1]
            first_char = token[0]
            if previous_char in ",;:" and first_char in ",;:.":
                score -= 0.16
            if previous_char in ".!?" and first_char.isalpha() and first_char.islower():
                score -= 0.10
            if previous_char.isalpha() and first_char in ",.!?;:":
                score += 0.03

        # --------------------------------------------------------------
        # Candidate should not repeatedly introduce the same word.
        # --------------------------------------------------------------

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

        if re.search(r"(?:^|\s)(?:\*\*|#+|\d+\.)", token):
            score -= 0.65

        if re.search(r"([^A-Za-z0-9\s])\1{2,}", token):
            score -= 0.65

        punctuation_count = len(re.findall(r"[^A-Za-z0-9\s]", token))
        if punctuation_count > max(3, len(token) // 3):
            score -= 0.65

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

    @staticmethod
    def _needs_completion(story: str) -> bool:

        stripped = story.strip()

        if not stripped:
            return True

        # A final word fragment is usually undesirable.
        last = stripped[-1]

        return last not in ".!?\"'”’"

    @staticmethod
    def _is_repetitive_continuation(story: str, token: str) -> bool:
        if not token or not token.strip():
            return True

        before_words = re.findall(r"[A-Za-z]+", story.lower())
        words = re.findall(r"[A-Za-z]+", (story + token).lower())

        if len(words) >= 2 and words[-1] == words[-2]:
            return True

        if len(words) >= 3:
            recent = words[-8:]
            for i in range(len(recent) - 2):
                if recent[i] == recent[i + 2] and recent[i] != recent[i + 1]:
                    return True

        if len(words) >= 6:
            before_phrases = [
                tuple(before_words[index:index + 3])
                for index in range(max(0, len(before_words) - 2))
            ]
            after_phrases = [
                tuple(words[index:index + 3])
                for index in range(len(words) - 2)
            ]
            before_repeats = sum(
                count - 1
                for count in Counter(before_phrases).values()
                if count > 1
            )
            after_repeats = sum(
                count - 1
                for count in Counter(after_phrases).values()
                if count > 1
            )
            if after_repeats > before_repeats:
                return True

        content_words = [
            word
            for word in words
            if len(word) >= 4 and word not in EmbedderLLM.STOPWORDS
        ]
        before_content_words = [
            word
            for word in before_words
            if len(word) >= 4 and word not in EmbedderLLM.STOPWORDS
        ]
        before_dominant = (
            max(Counter(before_content_words).values())
            if before_content_words
            else 0
        )
        if content_words and (
            max(Counter(content_words).values()) > before_dominant
            and max(Counter(content_words).values()) > 5
        ):
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

        content_words = [
            word
            for word in words
            if len(word) >= 4 and word not in cls.STOPWORDS
        ]
        dominant_content_word = (
            max(Counter(content_words).values())
            if content_words
            else 0
        )

        repeated_phrase_count = sum(
            count - 1
            for count in Counter(ngrams).values()
            if count > 1
        )

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
                r"\d+\s*[=+*/-]|[<>]|\*\*|#+|"
                r"(?:^|\s)\d+\.)",
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
                and dominant_content_word <= 5
                and repeated_phrase_count <= 2
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
        has_non_repetitive_candidate = any(
            candidate.token
            and not self._is_repetitive_continuation(story, candidate.token)
            for candidate in candidates
        )

        for candidate in candidates:

            self._embedding_stats["candidate_evaluations"] += 1

            token = candidate.token

            if not token:
                continue

            if re.search(r"(?:^|\s)(?:\*\*|#+|\d+\.)", token):
                continue

            if (
                has_non_repetitive_candidate
                and self._is_repetitive_continuation(story, token)
            ):
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

            if candidate.probability < self.MIN_CANDIDATE_PROBABILITY:
                rejection_reasons.append("candidate probability below minimum")
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

            score = self._score_candidate(
                story=story,
                token=token,
                probability=candidate.probability,
                topic=topic,
            )

            if naturalness < 0.25:
                rejection_reasons.append("naturalness score below 0.25")
                continue

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

    def _select_space_boundary_fallback(
        self,
        story: str,
        candidates,
        position: int,
        topic: str,
    ):
        """Rescue an otherwise unreachable fixed space boundary."""

        distance_to_target = position - len(story)
        if distance_to_target < 0:
            return None

        best = None
        best_score = float("-inf")

        for candidate in candidates:
            token = candidate.token
            if not token or len(token) <= distance_to_target:
                continue

            prefix = token[:distance_to_target]
            rescued_story = story + prefix + " "
            score = self._score_candidate(
                story=story,
                token=prefix + " ",
                probability=candidate.probability,
                topic=topic,
            )

            if score < 0.25:
                continue

            if score > best_score:
                best = rescued_story
                best_score = score

        return best

    def _select_character_boundary_fallback(
        self,
        story: str,
        candidates,
        character: str,
        position: int,
        topic: str,
    ):
        """Rescue a rare fixed character at an otherwise unreachable boundary."""

        distance_to_target = position - len(story)
        if distance_to_target < 0:
            return None

        best = None
        best_score = float("-inf")

        for candidate in candidates:
            token = candidate.token
            if not token or len(token) <= distance_to_target:
                continue

            prefix = token[:distance_to_target]
            if not prefix or prefix[-1].isalnum():
                continue
            if "\n" in prefix or "\r" in prefix:
                continue
            if re.search(r"([^A-Za-z0-9\s])\1{2,}", prefix):
                continue
            rescued_token = prefix + character
            score = self._score_candidate(
                story=story,
                token=rescued_token,
                probability=candidate.probability,
                topic=topic,
            )

            if score < 0.45:
                continue

            if score > best_score:
                best = story + rescued_token
                best_score = score

        return best

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

            # A wider pool is used only for the direct embedding decision.
            # Ordinary continuation still uses the conservative top-k pool,
            # preserving naturalness while rescuing rare target characters.
            distance_to_target = position - len(story)
            fallback_top_k = (
                self.SPACE_FALLBACK_TOP_K
                if character == " "
                else self.DIRECT_FALLBACK_TOP_K
            )
            if (
                top_k < fallback_top_k
                and distance_to_target <= self.DIRECT_FALLBACK_WINDOW
            ):
                fallback_candidates = (
                    self.llm_generator.get_next_token_candidates(
                        prompt=self._model_prompt(topic, story),
                        top_k=fallback_top_k,
                        temperature=temperature,
                    )
                )

                if character == " ":
                    selected = self._select_space_candidate(
                        story=story,
                        candidates=fallback_candidates,
                        position=position,
                        topic=topic,
                    )
                else:
                    selected = self._select_embedding_candidate(
                        story=story,
                        candidates=fallback_candidates,
                        character=character,
                        position=position,
                        topic=topic,
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

    def _generate_until_character(
        self,
        story: str,
        character: str,
        topic: str,
        temperature: float,
        top_k: int,
        max_steps: int,
    ):
        """Continue naturally until the required character occurs."""

        for _ in range(max_steps):
            candidates = self.llm_generator.get_next_token_candidates(
                prompt=self._model_prompt(topic, story),
                top_k=top_k,
                temperature=temperature,
            )
            if not candidates:
                break

            matching = []
            for candidate in candidates:
                token = candidate.token
                if not token or self._is_repetitive_continuation(story, token):
                    continue
                if re.search(r"(?:^|\s)(?:\*\*|#+|\d+\.)", token):
                    continue
                if character == " ":
                    found = any(char.isspace() for char in token)
                else:
                    found = character.lower() in token.lower()
                if found:
                    matching.append(candidate)

            if matching:
                selected = max(
                    matching,
                    key=lambda candidate: candidate.probability,
                )
                return story + selected.token

            valid = [
                candidate
                for candidate in candidates
                if candidate.token
                and not self._is_repetitive_continuation(story, candidate.token)
            ]
            normal = self._select_normal_candidate(
                story=story,
                candidates=valid,
                topic=topic,
            )
            if normal is None:
                break
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
            (0.75, 40), (0.75, 60), (0.75, 80),
            (0.80, 40), (0.80, 60), (0.80, 100),
        ]
        retry_instructions = (
            "",
            " Use a fresh narrative continuation and vary the wording.",
            " Take a different natural narrative path while staying on topic.",
        )

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
                retry_topic = topic + retry_instructions[(retry_round - 1) % len(retry_instructions)]
                retry_attempt_counter = [0]

                result = self._generate_to_position(
                    story=story,
                    character=character,
                    position=position,
                    topic=retry_topic,
                    temperature=retry_temperature,
                    top_k=retry_top_k,
                    max_steps=max_steps,
                    max_attempts=max_attempts,
                    attempt_counter=retry_attempt_counter,
                )
                attempt_counter[0] += retry_attempt_counter[0]

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

        total_attempt_counter = [0]
        self._embedding_stats = {
            "llm_calls": 0,
            "candidate_evaluations": 0,
            "retries": 0,
        }

        # Keep the caller's position list synchronized when adaptive
        # recovery has to record a natural occurrence instead of a fixed one.
        provided_positions = positions
        positions = list(positions)

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

            character_attempt_counter = [0]
            try:
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
                    attempt_counter=character_attempt_counter,
                )
            except RuntimeError as exc:
                logger.warning(
                    "Fixed position %d was unreachable for %r; "
                    "trying natural adaptive placement",
                    position,
                    character,
                )
                adaptive_story = self._generate_until_character(
                    story=story,
                    character=character,
                    topic=topic,
                    temperature=temperature,
                    top_k=max(top_k, 100),
                    max_steps=max_new_tokens * 8,
                )
                if adaptive_story is None:
                    raise exc
                start = len(story)
                target_index = next(
                    index
                    for index in range(start, len(adaptive_story))
                    if (
                        adaptive_story[index].isspace()
                        if character == " "
                        else adaptive_story[index].lower() == character.lower()
                    )
                )
                delta = target_index - position
                positions[index] = target_index
                position = target_index
                for following in range(index + 1, len(positions)):
                    positions[following] += delta
                story = adaptive_story
            total_attempt_counter[0] += character_attempt_counter[0]

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

        if isinstance(provided_positions, list):
            provided_positions[:] = positions

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
            attempts=total_attempt_counter[0],
        )


__all__ = [
    "EmbeddingResult",
    "EmbedderLLM",
]