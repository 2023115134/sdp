from app.evaluation.naturalness import summarize_cover_text


def test_naturalness_summary_is_data_driven():
    summary = summarize_cover_text("A quiet city. A quiet city.")
    assert summary["text_length"] == 27
    assert summary["sentence_count"] == 2
    assert summary["repeated_token_count"] > 0
