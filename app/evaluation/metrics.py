"""Metrics computed from actual embedding and extraction runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregate metrics for a collection of measured runs."""

    total_runs: int
    successful_runs: int
    extraction_correct: int
    embedded_characters: int
    correctly_extracted_characters: int
    total_embedding_seconds: float
    total_extraction_seconds: float
    total_candidate_evaluations: int

    @property
    def embedding_success_rate(self) -> float:
        return self.successful_runs / self.total_runs if self.total_runs else 0.0

    @property
    def extraction_accuracy(self) -> float:
        return self.extraction_correct / self.total_runs if self.total_runs else 0.0

    @property
    def character_extraction_accuracy(self) -> float:
        return (
            self.correctly_extracted_characters / self.embedded_characters
            if self.embedded_characters
            else 0.0
        )

    @property
    def average_embedding_seconds(self) -> float:
        return self.total_embedding_seconds / self.total_runs if self.total_runs else 0.0

    @property
    def average_extraction_seconds(self) -> float:
        return self.total_extraction_seconds / self.total_runs if self.total_runs else 0.0

    @property
    def average_candidate_evaluations_per_character(self) -> float:
        return (
            self.total_candidate_evaluations / self.embedded_characters
            if self.embedded_characters
            else 0.0
        )
