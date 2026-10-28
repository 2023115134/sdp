from app.evaluation.metrics import EvaluationMetrics
from app.evaluation.naturalness import summarize_cover_text


def test_naturalness_summary_is_data_driven():
    summary = summarize_cover_text("A quiet city. A quiet city.")
    assert summary["text_length"] == 27
    assert summary["sentence_count"] == 2
    assert summary["repeated_token_count"] > 0


def test_evaluation_metrics_use_successful_runs_and_actual_character_volume():
    rows = [
        {"success": True, "message_size_chars": 10, "llm_embedding_ms": 2000.0, "extraction_ms": 1000.0, "candidate_evaluations": 10, "correctly_extracted_characters": 10, "embedded_characters": 10},
        {"success": False, "message_size_chars": 10, "llm_embedding_ms": 5000.0, "extraction_ms": 2000.0, "candidate_evaluations": 50, "correctly_extracted_characters": 0, "embedded_characters": 10},
    ]

    metrics = EvaluationMetrics.from_run_results(rows)

    assert metrics.total_runs == 2
    assert metrics.successful_runs == 1
    assert metrics.embedding_success_rate == 0.5
    assert metrics.extraction_accuracy == 1.0
    assert metrics.average_embedding_seconds == 2.0
    assert metrics.average_extraction_seconds == 1.0
    assert metrics.average_candidate_evaluations_per_character == 1.0
