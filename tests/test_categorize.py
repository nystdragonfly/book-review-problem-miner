import numpy as np

from problem_miner.categorize import _NEGATION_RE, categorize_clusters
from problem_miner.config import PipelineConfig


def test_negation_regex_matches_common_negation_words():
    for text in [
        "I didn't expect to love this.",
        "Not for me.",
        "It wasn't what I wanted.",
        "No stat screens here.",
        "I never finished it.",
    ]:
        assert _NEGATION_RE.search(text), f"expected a match in: {text!r}"


def test_negation_regex_does_not_match_plain_positive_text():
    assert _NEGATION_RE.search("I loved this book, it was wonderful.") is None


def test_categorize_clusters_assigns_dominant_sentiment_and_orders_by_similarity():
    sentences = [
        "This is a fairly long negative sentence about the plot.",
        "This is another long negative sentence about pacing issues.",
        "This is a long positive sentence about characters.",
    ]
    # Three vectors that are mutually close to each other (same rough
    # direction) so they form one obvious cluster.
    embeddings = np.array([
        [1.0, 0.1],
        [0.9, 0.2],
        [0.8, 0.3],
    ])
    cluster_labels = np.array([0, 0, 0])
    sentiment_labels = ["negative", "negative", "positive"]

    result = categorize_clusters(sentences, embeddings, cluster_labels, sentiment_labels)

    assert len(result) == 1
    cluster = result[0]
    assert cluster.category == "negative"  # 2/3 negative -> dominant
    assert cluster.sentiment_breakdown == {"negative": 2, "positive": 1}
    assert set(cluster.member_indices_ranked) == {0, 1, 2}
    # similarities should be sorted descending (most representative first)
    assert cluster.similarities == sorted(cluster.similarities, reverse=True)


def test_categorize_clusters_flags_junk_by_avg_word_count_not_sentiment():
    # Short bare interjections -- avg words well under the junk threshold
    # -- should be "junk" even though their sentiment is positive, and
    # even though a separate real cluster with the same sentiment isn't.
    sentences = ["No.", "Wow.", "This is a proper long positive sentence about the book."]
    embeddings = np.array([
        [0.1, 1.0],
        [0.2, 0.9],
        [1.0, 0.1],
    ])
    cluster_labels = np.array([1, 1, 0])  # cluster 0: junk pair, cluster 1: real sentence
    sentiment_labels = ["positive", "positive", "positive"]

    result = categorize_clusters(sentences, embeddings, cluster_labels, sentiment_labels)
    by_id = {c.cluster_id: c for c in result}

    assert by_id[1].category == "junk"
    assert by_id[0].category == "positive"


def test_categorize_clusters_excludes_noise_points():
    sentences = ["a", "b", "c"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    cluster_labels = np.array([0, -1, 0])  # index 1 is noise
    sentiment_labels = ["positive", "negative", "positive"]

    result = categorize_clusters(sentences, embeddings, cluster_labels, sentiment_labels)
    assert len(result) == 1
    assert 1 not in result[0].member_indices_ranked


def test_categorize_clusters_respects_custom_junk_threshold():
    sentences = ["one two three four five", "six seven eight nine ten"]
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1]])
    cluster_labels = np.array([0, 0])
    sentiment_labels = ["negative", "negative"]

    strict_config = PipelineConfig(junk_avg_word_threshold=10.0)  # 5-word sentences now count as junk
    result = categorize_clusters(sentences, embeddings, cluster_labels, sentiment_labels, strict_config)
    assert result[0].category == "junk"
