"""
End-to-end orchestration: source -> clean -> split -> embed -> cluster ->
categorize -> label -> PipelineResults.

This is the one place that wires every other module together. Individual
stages stay independently testable/usable (e.g. cluster.py doesn't know
labeling exists) -- run_pipeline() is just the composition.
"""
import re

from .categorize import categorize_clusters, classify_sentiment
from .clean import clean_reviews
from .cluster import cluster_embeddings
from .config import DEFAULT_CONFIG, PipelineConfig
from .embed import embed_sentences
from .label import label_cluster
from .results import ClusterResult, PipelineResults, SentenceRecord
from .sources.base import ReviewSource
from .split import split_into_sentences


def _safe_cache_key(source_name: str, book_id: str) -> str:
    raw = f"{source_name}-{book_id}"
    return re.sub(r"[^A-Za-z0-9_-]", "_", raw)


def run_pipeline(
    source: ReviewSource,
    book_id: str,
    book_context: str | None = None,
    config: PipelineConfig = DEFAULT_CONFIG,
    n_examples_per_cluster: int | None = None,
) -> PipelineResults:
    """Run the full pipeline for one book/source and return structured
    results (see results.py -- no ranking/"top N" applied here).

    n_examples_per_cluster: if set, only the top-K most representative
    sentences per cluster (by similarity to the cluster center) are kept
    in the output instead of the full membership. Full membership by
    default -- a future consumer may want to browse every sentence, not
    just a pre-curated slice (see results.py docstring).
    """
    raw_reviews = source.load(book_id)
    cleaned = clean_reviews(raw_reviews, config)
    sentences = split_into_sentences(cleaned)
    sentence_texts = [s.text for s in sentences]

    cache_key = _safe_cache_key(source.name, book_id)
    embeddings = embed_sentences(sentence_texts, cache_key, config)

    cluster_labels = cluster_embeddings(embeddings, config)
    noise_count = int((cluster_labels == -1).sum())

    sentiment_labels = classify_sentiment(sentence_texts, config)
    cluster_members = categorize_clusters(
        sentence_texts, embeddings, cluster_labels, sentiment_labels, config,
    )

    clusters = []
    for cm in cluster_members:
        indices = cm.member_indices_ranked
        sims = cm.similarities
        if n_examples_per_cluster is not None:
            indices = indices[:n_examples_per_cluster]
            sims = sims[:n_examples_per_cluster]

        sentence_records = [
            SentenceRecord(
                text=sentences[i].text,
                review_id=sentences[i].review_id,
                rating=sentences[i].rating,
                sentiment=sentiment_labels[i],
                similarity_to_cluster_center=float(sim),
            )
            for i, sim in zip(indices, sims)
        ]

        if cm.category == "junk":
            # Don't ask the LLM to find a theme in content that by
            # definition has none -- it will confidently hallucinate one.
            # Concrete example hit during development: a cluster of bare
            # numbers ("2.", "3." -- orphaned numbered-list markers from
            # sentence-splitting) got labeled "Unresponsive Customer
            # Service" (see devlog 2026-08-20).
            title = "Low-content fragments (no real theme)"
            summary = "Short interjections, bare numbers, or other fragments with no coherent theme to summarize."
        else:
            # Labeling only ever needs a handful of examples, regardless
            # of n_examples_per_cluster -- use the top 5 by similarity.
            title_examples = [sentences[i].text for i in cm.member_indices_ranked[:5]]
            title, summary = label_cluster(title_examples, cm.category, book_context, config)

        clusters.append(ClusterResult(
            cluster_id=cm.cluster_id,
            category=cm.category,
            size=len(cm.member_indices_ranked),
            avg_words_per_sentence=cm.avg_words,
            sentiment_breakdown=cm.sentiment_breakdown,
            title=title,
            summary=summary,
            sentences=sentence_records,
        ))

    return PipelineResults.create(
        book_id=book_id,
        source_name=source.name,
        total_raw_reviews=len(raw_reviews),
        cleaned_reviews=len(cleaned),
        total_sentences=len(sentences),
        noise_sentence_count=noise_count,
        clusters=clusters,
    )
