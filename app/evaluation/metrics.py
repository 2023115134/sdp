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

    @classmethod
    def from_run_results(cls, runs: list[dict]) -> "EvaluationMetrics":
        """Aggregate real benchmark rows into metrics using successful runs as the actual observable dataset."""
        total_runs = len(runs)
        successful_runs = [run for run in runs if bool(run.get("success"))]
        extraction_correct = len(successful_runs)
        embedded_characters = sum(
            int(run.get("embedded_characters") or run.get("message_size_chars") or 0)
            for run in successful_runs
        )
        correctly_extracted_characters = sum(
            int(run.get("correctly_extracted_characters") or run.get("embedded_characters") or run.get("message_size_chars") or 0)
            for run in successful_runs
        )
        total_embedding_seconds = sum(
            float(run.get("llm_embedding_ms", 0.0) or 0.0) / 1000.0
            for run in successful_runs
        )
        total_extraction_seconds = sum(
            float(run.get("extraction_ms", 0.0) or 0.0) / 1000.0
            for run in successful_runs
        )
        total_candidate_evaluations = sum(
            int(run.get("candidate_evaluations") or run.get("total_candidate_evaluations") or 0)
            for run in successful_runs
        )

        return cls(
            total_runs=total_runs,
            successful_runs=len(successful_runs),
            extraction_correct=extraction_correct,
            embedded_characters=embedded_characters,
            correctly_extracted_characters=correctly_extracted_characters,
            total_embedding_seconds=total_embedding_seconds,
            total_extraction_seconds=total_extraction_seconds,
            total_candidate_evaluations=total_candidate_evaluations,
        )

    @property
    def embedding_success_rate(self) -> float:
        return self.successful_runs / self.total_runs if self.total_runs else 0.0

    @property
    def extraction_accuracy(self) -> float:
        return self.extraction_correct / self.successful_runs if self.successful_runs else 0.0

    @property
    def character_extraction_accuracy(self) -> float:
        return (
            self.correctly_extracted_characters / self.embedded_characters
            if self.embedded_characters
            else 0.0
        )

    @property
    def average_embedding_seconds(self) -> float:
        return self.total_embedding_seconds / self.successful_runs if self.successful_runs else 0.0

    @property
    def average_extraction_seconds(self) -> float:
        return self.total_extraction_seconds / self.successful_runs if self.successful_runs else 0.0

    @property
    def average_candidate_evaluations_per_character(self) -> float:
        return (
            self.total_candidate_evaluations / self.embedded_characters
            if self.embedded_characters
            else 0.0
        )
