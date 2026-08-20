from problem_miner.results import ClusterResult, PipelineResults, SentenceRecord


def make_results() -> PipelineResults:
    sentences = [
        SentenceRecord(
            text="Didn't like the artwork.", review_id="r1", rating=2,
            sentiment="negative", similarity_to_cluster_center=0.91,
        ),
        SentenceRecord(
            text="The art felt flat.", review_id="r2", rating=3,
            sentiment="negative", similarity_to_cluster_center=0.85,
        ),
    ]
    cluster = ClusterResult(
        cluster_id=0, category="negative", size=2, avg_words_per_sentence=4.5,
        sentiment_breakdown={"negative": 2}, title="Artwork Complaints",
        summary="Readers disliked the artwork.", sentences=sentences,
    )
    return PipelineResults.create(
        book_id="test-book", source_name="test-source",
        total_raw_reviews=10, cleaned_reviews=8, total_sentences=20,
        noise_sentence_count=5, clusters=[cluster],
    )


def test_create_sets_generated_at_automatically():
    results = make_results()
    assert results.generated_at  # non-empty, set by create()
    assert "T" in results.generated_at  # looks like an ISO 8601 timestamp


def test_save_and_load_json_round_trips_exactly(tmp_path):
    results = make_results()
    out_path = tmp_path / "results.json"
    results.save_json(out_path)
    reloaded = PipelineResults.load_json(out_path)
    assert reloaded == results


def test_save_json_accepts_plain_string_path_not_just_path_object(tmp_path):
    # Real bug hit during development: save_json originally required a
    # Path object and crashed on a plain string. Regression test.
    results = make_results()
    out_path_str = str(tmp_path / "results.json")
    results.save_json(out_path_str)  # must not raise
    reloaded = PipelineResults.load_json(out_path_str)
    assert reloaded == results


def test_save_json_creates_parent_directories(tmp_path):
    results = make_results()
    nested_path = tmp_path / "nested" / "dir" / "results.json"
    results.save_json(nested_path)
    assert nested_path.exists()
