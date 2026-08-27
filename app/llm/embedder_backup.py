"""
EmbedderLLM - Phase 1 paper-based implementation.

Based on the EmbedderLLM construction described in:

"An LLM Framework For Cryptography Over Chat Channels"

Core idea:

    TOPIC + Story
          |
          v
    GPT-2 top-k candidates
          |
          v
    candidate set Y
          |
          v
    Does candidate place C_i at b_i?
       /              \
     YES               NO
      |                 |
   accept          retry / adjust
      |
      v
   next C_i

Important:
- The secret is NEVER directly inserted into the Story.
- Characters are embedded only through LLM-generated candidate tokens.
- The target condition is:

      uppercase(Story[b_i]) == C_i

- T starts at 0.7.
- k starts at 40.
- The paper discusses T in [0.7, 0.9]
  and k in [40, 60].
- On unsuccessful retries, the Story is rolled back and
  k is increased.
"""

from __future__ import annotations

import logging
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
    """
    Result returned by EmbedderLLM.
    """

    story: str
    embedded_characters: str
    positions: list[int]
    attempts: int


# ======================================================================
# EMBEDDER
# ======================================================================

class EmbedderLLM:
    """
    Paper-inspired EmbedderLLM implementation.

    The implementation follows the sequential structure of Algorithm 1:

    1. Start from Story_0.
    2. Consider C_i and b_i.
    3. Obtain top-k next-token candidates.
    4. If Story + candidate is still before b_i,
       select a normal high-probability candidate.
    5. When approaching b_i, search for a candidate y such that:

           Char(Story || y, b_i) == C_i

    6. If no candidate works, retry.
    7. If retries are exhausted, rollback and increase k.
    8. Continue for the next character.
    """

    # Paper starting parameters
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_K = 40

    # Paper's discussed useful ranges
    MAX_TEMPERATURE = 0.9
    MAX_TOP_K = 60

    # Default retry configuration
    DEFAULT_RETRIES = 20

    # Small safety limit so an accidental infinite loop cannot occur
    DEFAULT_MAX_GENERATION_STEPS = 10000

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
            raise TypeError(
                "topic must be a string"
            )

        if not topic.strip():
            raise ValueError(
                "topic must not be empty"
            )

        if not isinstance(characters, str):
            raise TypeError(
                "characters must be a string"
            )

        if not characters:
            raise ValueError(
                "characters must not be empty"
            )

        if positions is None:
            raise ValueError(
                "positions must not be None"
            )

        if len(characters) != len(positions):
            raise ValueError(
                "Number of characters must equal "
                "number of positions."
            )

        previous = -1

        for position in positions:

            if not isinstance(position, int):
                raise TypeError(
                    "All positions must be integers."
                )

            if position < 0:
                raise ValueError(
                    "Positions cannot be negative."
                )

            if position <= previous:
                raise ValueError(
                    "Positions must be strictly increasing."
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
        """
        The paper treats uppercase/lowercase letters equivalently.
        """

        if not actual:
            return False

        if not required:
            return False

        return (
            actual.upper()
            == required.upper()
        )

    # ==================================================================
    # CHECK TARGET POSITION
    # ==================================================================

    @classmethod
    def _candidate_hits_position(
        cls,
        story: str,
        token: str,
        target_position: int,
        required_character: str,
    ) -> bool:
        """
        Check whether Story || token places the required character
        exactly at target_position.

        This is the core paper condition:

            Char(Story || y, b_i) = C_i
        """

        if not token:
            return False

        new_story = story + token

        if target_position >= len(new_story):
            return False

        actual = new_story[target_position]

        return cls._character_matches(
            actual,
            required_character,
        )

    # ==================================================================
    # CHECK WHETHER POSITION HAS BEEN CROSSED
    # ==================================================================

    @staticmethod
    def _crosses_position(
        story: str,
        token: str,
        position: int,
    ) -> bool:
        """
        Return True if Story || token reaches or crosses position.
        """

        return (
            len(story + token)
            > position
        )

    # ==================================================================
    # CHECK WHETHER ALL PREVIOUS TARGETS ARE VALID
    # ==================================================================

    @classmethod
    def _previous_positions_valid(
        cls,
        story: str,
        characters: str,
        positions: Sequence[int],
        current_index: int,
    ) -> bool:
        """
        Verify positions that have already been embedded.
        """

        for index in range(current_index):

            position = positions[index]
            required = characters[index]

            if position >= len(story):
                return False

            actual = story[position]

            if not cls._character_matches(
                actual,
                required,
            ):
                return False

        return True

    # ==================================================================
    # SELECT NORMAL CANDIDATE
    # ==================================================================

    @staticmethod
    def _select_normal_candidate(
        candidates,
    ):
        """
        Select the highest-probability candidate.

        The generator already returns candidates ordered by probability.
        """

        if not candidates:
            return None

        return candidates[0]

    # ==================================================================
    # FIND EMBEDDING CANDIDATE
    # ==================================================================

    @classmethod
    def _find_embedding_candidate(
        cls,
        story: str,
        candidates,
        target_position: int,
        required_character: str,
    ):
        """
        Search Y:

            Y = {
                y in Y_top-k |
                Char(Story || y, b_i) == C_i
            }

        Returns the first/highest-probability valid candidate.
        """

        for candidate in candidates:

            token = candidate.token

            if not token:
                continue

            if cls._candidate_hits_position(
                story=story,
                token=token,
                target_position=target_position,
                required_character=required_character,
            ):
                return candidate

        return None

    # ==================================================================
    # GENERATE TOWARD TARGET
    # ==================================================================

    def _generate_until_target(
        self,
        story: str,
        target_position: int,
        characters: str,
        positions: Sequence[int],
        current_index: int,
        temperature: float,
        top_k: int,
        retry_limit: int,
        total_attempts: list[int],
        max_generation_steps: int,
    ):
        """
        Generate ordinary LLM text until the next target position
        is approached.

        At every step:

            candidates = LLM(TOPIC, Story, T, k)

        If the highest probability candidate stays safely before
        the target, it is accepted.

        Once candidates can reach the target position, we stop and
        let the embedding search choose the required candidate.
        """

        steps = 0

        while len(story) <= target_position:

            steps += 1

            if steps > max_generation_steps:
                raise RuntimeError(
                    "Maximum generation steps exceeded while "
                    f"approaching position {target_position}."
                )

            candidates = (
                self.llm_generator
                .get_next_token_candidates(
                    prompt=story,
                    top_k=top_k,
                    temperature=temperature,
                )
            )

            if not candidates:
                raise RuntimeError(
                    "LLM returned no candidates."
                )

            # ----------------------------------------------------------
            # Search immediately if ANY candidate can reach target.
            # ----------------------------------------------------------

            embedding_candidate = (
                self._find_embedding_candidate(
                    story=story,
                    candidates=candidates,
                    target_position=target_position,
                    required_character=characters[current_index],
                )
            )

            if embedding_candidate is not None:

                return (
                    story,
                    embedding_candidate,
                    True,
                )

            # ----------------------------------------------------------
            # Determine whether the top candidate is still safely
            # before the target.
            # ----------------------------------------------------------

            selected = self._select_normal_candidate(
                candidates
            )

            if selected is None:
                raise RuntimeError(
                    "Unable to select a normal candidate."
                )

            token = selected.token

            if not token:
                continue

            total_attempts[0] += len(candidates)

            # ----------------------------------------------------------
            # If token would cross the target, we cannot blindly accept
            # it because that would permanently determine Story[b_i].
            # ----------------------------------------------------------

            if self._crosses_position(
                story=story,
                token=token,
                position=target_position,
            ):

                return (
                    story,
                    None,
                    False,
                )

            # ----------------------------------------------------------
            # Token stays before target.
            # ----------------------------------------------------------

            story += token

            # ----------------------------------------------------------
            # Safety check: previously embedded positions must remain
            # unchanged.
            # ----------------------------------------------------------

            if not self._previous_positions_valid(
                story=story,
                characters=characters,
                positions=positions,
                current_index=current_index,
            ):
                raise RuntimeError(
                    "Previously embedded position became invalid."
                )

        return (
            story,
            None,
            False,
        )

    # ==================================================================
    # EMBED ONE CHARACTER
    # ==================================================================

    def _embed_character(
        self,
        story: str,
        character: str,
        position: int,
        characters: str,
        positions: Sequence[int],
        current_index: int,
        temperature: float,
        top_k: int,
        max_retries: int,
        total_attempts: list[int],
        max_generation_steps: int,
    ):
        """
        Embed one character at one exact position.

        This method implements the important retry/rollback behavior.
        """

        original_story = story

        retry = 0

        current_temperature = temperature
        current_top_k = top_k

        while retry < max_retries:

            retry += 1

            logger.info(
                "Character %r target=%d retry=%d "
                "T=%.2f k=%d",
                character,
                position,
                retry,
                current_temperature,
                current_top_k,
            )

            # ==========================================================
            # STEP 1
            # Generate ordinary text until we are close to b_i.
            # ==========================================================

            working_story = original_story

            (
                working_story,
                candidate,
                found_during_generation,
            ) = self._generate_until_target(
                story=working_story,
                target_position=position,
                characters=characters,
                positions=positions,
                current_index=current_index,
                temperature=current_temperature,
                top_k=current_top_k,
                retry_limit=max_retries,
                total_attempts=total_attempts,
                max_generation_steps=max_generation_steps,
            )

            # ==========================================================
            # Candidate was found while approaching the target.
            # ==========================================================

            if found_during_generation and candidate is not None:

                token = candidate.token

                new_story = (
                    working_story
                    + token
                )

                # ------------------------------------------------------
                # Final exact check
                # ------------------------------------------------------

                if (
                    position < len(new_story)
                    and self._character_matches(
                        new_story[position],
                        character,
                    )
                ):

                    logger.info(
                        "VALID candidate: token=%r "
                        "probability=%.6f "
                        "target=%d character=%r",
                        token,
                        candidate.probability,
                        position,
                        character,
                    )

                    return (
                        new_story,
                        retry,
                    )

            # ==========================================================
            # STEP 2
            # Explicitly request candidates at the current position.
            # ==========================================================

            candidates = (
                self.llm_generator
                .get_next_token_candidates(
                    prompt=working_story,
                    top_k=current_top_k,
                    temperature=current_temperature,
                )
            )

            total_attempts[0] += len(candidates)

            candidate = (
                self._find_embedding_candidate(
                    story=working_story,
                    candidates=candidates,
                    target_position=position,
                    required_character=character,
                )
            )

            if candidate is not None:

                new_story = (
                    working_story
                    + candidate.token
                )

                if (
                    position < len(new_story)
                    and self._character_matches(
                        new_story[position],
                        character,
                    )
                ):

                    logger.info(
                        "VALID candidate found: "
                        "token=%r probability=%.6f "
                        "target=%d character=%r",
                        candidate.token,
                        candidate.probability,
                        position,
                        character,
                    )

                    return (
                        new_story,
                        retry,
                    )

            logger.info(
                "No valid candidate at target=%d "
                "for character=%r",
                position,
                character,
            )

            # ==========================================================
            # STEP 3
            # Increase temperature within paper's useful range.
            # ==========================================================

            if current_temperature < self.MAX_TEMPERATURE:

                current_temperature = min(
                    self.MAX_TEMPERATURE,
                    round(
                        current_temperature + 0.05,
                        2,
                    ),
                )

                logger.info(
                    "Retrying with increased temperature: %.2f",
                    current_temperature,
                )

                continue

            # ==========================================================
            # STEP 4
            # Temperature has reached 0.9.
            #
            # Paper recovery:
            # rollback Story to previous position and increase k.
            # ==========================================================

            if current_top_k < self.MAX_TOP_K:

                current_top_k += 1

                # Rollback to the story before this target.
                working_story = original_story

                logger.info(
                    "Temperature limit reached. "
                    "Increasing top-k to %d and retrying.",
                    current_top_k,
                )

                continue

            # ==========================================================
            # STEP 5
            # Restart with the original parameters if necessary.
            #
            # This is a practical safeguard because GPT-2 can have
            # local candidate limitations.
            # ==========================================================

            current_temperature = temperature
            current_top_k = top_k

        raise RuntimeError(
            "Unable to embed character "
            f"{character!r} at position {position}. "
            f"Retries={max_retries}, "
            f"initial_temperature={temperature}, "
            f"initial_top_k={top_k}, "
            f"attempts={total_attempts[0]}"
        )

    def _embed_character_with_beam(
        self,
        story: str,
        character: str,
        position: int,
        temperature: float,
        top_k: int,
        max_retries: int,
        total_attempts: list[int],
        max_attempts: int,
        max_generation_steps: int,
        beam_width: int = 16,
    ) -> tuple[str, int]:
        """Explore alternate LLM token paths before retrying parameters."""

        retry_temperature = temperature
        retry_top_k = top_k

        for retry in range(1, max_retries + 1):
            beam: list[tuple[str, float]] = [(story, 0.0)]

            for _ in range(max_generation_steps):
                next_beam: list[tuple[str, float]] = []

                for candidate_story, score in beam:
                    if len(candidate_story) > position:
                        continue

                    candidates = self.llm_generator.get_next_token_candidates(
                        prompt=candidate_story,
                        top_k=retry_top_k,
                        temperature=retry_temperature,
                    )
                    total_attempts[0] += len(candidates)

                    if total_attempts[0] > max_attempts:
                        raise RuntimeError(
                            "Maximum candidate evaluations exceeded while "
                            f"embedding character {character!r}."
                        )

                    for candidate in candidates:
                        token = candidate.token
                        if not token:
                            continue

                        new_story = candidate_story + token
                        new_score = score + max(
                            float(candidate.probability),
                            1e-12,
                        )

                        if len(new_story) > position:
                            if self._character_matches(
                                new_story[position],
                                character,
                            ):
                                logger.info(
                                    "VALID candidate: token=%r "
                                    "probability=%.6f target=%d "
                                    "character=%r",
                                    token,
                                    candidate.probability,
                                    position,
                                    character,
                                )
                                return new_story, retry
                        else:
                            next_beam.append((new_story, new_score))

                if not next_beam:
                    break

                next_beam.sort(key=lambda item: item[1], reverse=True)
                beam = next_beam[:beam_width]

            if retry_temperature < self.MAX_TEMPERATURE:
                retry_temperature = min(
                    self.MAX_TEMPERATURE,
                    round(retry_temperature + 0.05, 2),
                )
            elif retry_top_k < self.MAX_TOP_K:
                retry_top_k += 1
                retry_temperature = temperature
            else:
                retry_temperature = temperature
                retry_top_k = top_k

        raise RuntimeError(
            "Unable to embed character "
            f"{character!r} at position {position} after "
            f"{max_retries} retries and {total_attempts[0]} attempts."
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
        """
        Embed the supplied character sequence into LLM-generated text.

        Parameters
        ----------
        topic:
            Initial topic/context.

        characters:
            Mapped character sequence C.

        positions:
            Increasing embedding positions b.

        initial_story:
            Optional initial Story_0.

        temperature:
            Paper starting temperature: 0.7.

        top_k:
            Paper starting top-k: 40.

        max_new_tokens:
            Safety limit for generation.

        max_attempts:
            Maximum total candidate evaluations.

        max_retries:
            Retry count for each target character.
        """

        # --------------------------------------------------------------
        # Validate
        # --------------------------------------------------------------

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
        # Initial Story
        # --------------------------------------------------------------

        if (
            initial_story
            and initial_story.strip()
        ):
            story = initial_story.strip()
        else:
            story = topic.strip()

        # --------------------------------------------------------------
        # Verify that initial Story doesn't already conflict with
        # target positions.
        # --------------------------------------------------------------

        for character, position in zip(
            characters,
            positions,
        ):

            if position < len(story):

                actual = story[position]

                if not self._character_matches(
                    actual,
                    character,
                ):
                    raise ValueError(
                        "Initial story conflicts with embedding "
                        f"position {position}: "
                        f"actual={actual!r}, "
                        f"required={character!r}"
                    )

        # --------------------------------------------------------------
        # Total attempts stored in mutable list so helper functions
        # can update it.
        # --------------------------------------------------------------

        total_attempts = [0]

        logger.info("=" * 70)
        logger.info(
            "STARTING PAPER-BASED EmbedderLLM"
        )
        logger.info(
            "Model: GPT-2"
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
            "Initial story length: %d",
            len(story),
        )
        logger.info(
            "Initial temperature: %.2f",
            temperature,
        )
        logger.info(
            "Initial top-k: %d",
            top_k,
        )
        logger.info("-" * 70)

        # --------------------------------------------------------------
        # Embed C_0, C_1, ..., C_n-1 sequentially.
        # --------------------------------------------------------------

        for index, (
            character,
            position,
        ) in enumerate(
            zip(characters, positions)
        ):

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

            # ----------------------------------------------------------
            # Safety check
            # ----------------------------------------------------------

            if total_attempts[0] >= max_attempts:

                raise RuntimeError(
                    "Maximum candidate evaluations exceeded "
                    f"before character {character!r}."
                )

            # ----------------------------------------------------------
            # Save previous successful story position.
            #
            # This corresponds to the paper's rollback idea.
            # ----------------------------------------------------------

            previous_successful_story = story

            # ----------------------------------------------------------
            # Embed character
            # ----------------------------------------------------------

            (
                story,
                retries_used,
            ) = self._embed_character_with_beam(
                story=previous_successful_story,
                character=character,
                position=position,
                temperature=temperature,
                top_k=top_k,
                max_retries=max_retries,
                total_attempts=total_attempts,
                max_attempts=max_attempts,
                max_generation_steps=max_new_tokens * 20,
            )

            # ----------------------------------------------------------
            # Verify immediately
            # ----------------------------------------------------------

            if position >= len(story):

                raise RuntimeError(
                    "Embedding produced a story that does not "
                    f"reach position {position}."
                )

            actual = story[position]

            if not self._character_matches(
                actual,
                character,
            ):

                raise RuntimeError(
                    "Embedding validation failed: "
                    f"position={position}, "
                    f"expected={character!r}, "
                    f"actual={actual!r}"
                )

            logger.info(
                "Character %r successfully embedded "
                "at position %d.",
                character,
                position,
            )

        # ==============================================================
        # FINAL VALIDATION
        # ==============================================================

        logger.info("-" * 70)
        logger.info(
            "FINAL EMBEDDING VALIDATION"
        )

        for character, position in zip(
            characters,
            positions,
        ):

            if position >= len(story):

                raise RuntimeError(
                    "Final validation failed: "
                    f"position {position} is outside "
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

        logger.info(
            "Embedding completed successfully."
        )

        logger.info(
            "Final story length: %d",
            len(story),
        )

        logger.info(
            "Total candidate evaluations: %d",
            total_attempts[0],
        )

        return EmbeddingResult(
            story=story,
            embedded_characters=characters,
            positions=list(positions),
            attempts=total_attempts[0],
        )


# ======================================================================
# TEST
# ======================================================================

def _run_test() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    print("=" * 70)
    print("EMBEDDER LLM TEST")
    print("=" * 70)

    generator = LLMGenerator()

    embedder = EmbedderLLM(
        llm_generator=generator
    )

    topic = (
        "A student is walking through "
        "a beautiful city"
    )

    characters = "E"

    positions = [50]

    print("\nTopic:")
    print(topic)

    print("\nCharacter:")
    print(characters)

    print("\nPosition:")
    print(positions)

    print("\nGenerating cover text...")

    result = embedder.embed(
        topic=topic,
        characters=characters,
        positions=positions,
        temperature=0.7,
        top_k=40,
        max_new_tokens=128,
        max_attempts=10000,
        max_retries=20,
    )

    print("\n" + "=" * 70)
    print("EMBEDDING SUCCESS")
    print("=" * 70)

    print("\nCover text:")
    print(result.story)

    print("\nPositions:")
    print(result.positions)

    print("\nExpected:")
    print(result.embedded_characters)

    print("\nActual:")

    for position in result.positions:
        print(result.story[position])

    print("\nAttempts:")
    print(result.attempts)


if __name__ == "__main__":
    _run_test()


__all__ = [
    "EmbeddingResult",
    "EmbedderLLM",
]