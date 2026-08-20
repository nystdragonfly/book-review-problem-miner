"""
Per-sentence sentiment classification + per-cluster categorization
(negative / positive / neutral / junk).

Sentiment is classified from each sentence's own text, not the source
review's star rating -- a 3-star review contains both praise and
complaint sentences, so the review's rating is a noisy proxy for any one
sentence. Junk is detected via average word count per *cluster* (bare
interjections/numbers cluster together), not a per-sentence length
cutoff -- see CLAUDE.md "Where things stand" for the empirical threshold
(junk clusters averaged 1.3-2.4 words/sentence; the next-lowest
legitimate cluster averaged 5.0).
"""
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_CONFIG, PipelineConfig

# Matches common negation markers/contractions. Used to flag candidates
# for the negation re-check below -- deliberately broad/inclusive (a
# false-positive match just means one extra LLM call, not an error).
_NEGATION_RE = re.compile(
    r"\b(not|no|never|n't|didn|dont|wasn|wasnt|isn|isnt|won|wont|doesn|doesnt)\b",
    re.IGNORECASE,
)


def _recheck_negation_candidates(
    sentences: list[str], sentiment_labels: list[str], config: PipelineConfig,
) -> list[str]:
    """Give a general-purpose local LLM a second opinion on sentences the
    small classifier called negative that also contain a negation marker.

    Found via manual investigation (devlog 2026-08-20): the small
    classifier (trained on tweets) systematically misreads negation used
    rhetorically for praise -- "scratched an itch I didn't know I had",
    "no stat screens, no XP bars, just a genuinely strange meditation" --
    as negative, because it keys on the negation word rather than the
    sentence's actual meaning.

    Only the negative->positive disagreement direction is applied here.
    negative->neutral disagreements are deliberately left alone: spot-
    checking that bucket showed a real mix of genuine corrections
    (compound/hedged sentences like "Good, not great.") and real
    regressions (terse dismissals like "Not for me." wrongly softened to
    neutral) -- not safe to auto-apply without per-sentence judgment,
    which defeats the point of automating this at all.
    """
    import requests

    updated = list(sentiment_labels)
    for i, (text, label) in enumerate(zip(sentences, sentiment_labels)):
        if label != "negative" or not _NEGATION_RE.search(text):
            continue
        prompt = (
            "This is a single sentence extracted from a reader's review of a book. "
            "Classify its sentiment as exactly one word: positive, negative, or neutral. "
            "Respond with ONLY that single word.\n\n"
            f'Sentence: "{text}"'
        )
        resp = requests.post(config.ollama_url, json={
            "model": config.ollama_model, "prompt": prompt, "stream": False,
        }, timeout=config.ollama_timeout_seconds)
        resp.raise_for_status()
        llm_label = resp.json()["response"].strip().lower()
        if "positive" in llm_label:
            updated[i] = "positive"
    return updated


def classify_sentiment(sentences: list[str], config: PipelineConfig = DEFAULT_CONFIG) -> list[str]:
    from transformers import pipeline
    clf = pipeline("sentiment-analysis", model=config.sentiment_model, device=0, truncation=True)
    results = clf(sentences, batch_size=config.sentiment_batch_size)
    labels = [r["label"] for r in results]
    if config.enable_negation_recheck:
        labels = _recheck_negation_candidates(sentences, labels, config)
    return labels


@dataclass
class ClusterMembers:
    cluster_id: int
    category: str  # "negative" | "positive" | "neutral" | "junk"
    sentiment_breakdown: dict[str, int]
    avg_words: float
    # Both lists are parallel and ordered most-representative-first
    # (highest cosine similarity to the cluster's own center).
    member_indices_ranked: list[int]
    similarities: list[float]


def categorize_clusters(
    sentences: list[str],
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    sentiment_labels: list[str],
    config: PipelineConfig = DEFAULT_CONFIG,
) -> list[ClusterMembers]:
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    cluster_ids = sorted(set(cluster_labels) - {-1})

    results = []
    for cid in cluster_ids:
        member_idx = np.where(cluster_labels == cid)[0]
        member_sentiments = [sentiment_labels[i] for i in member_idx]
        dominant, _ = Counter(member_sentiments).most_common(1)[0]
        avg_words = float(np.mean([len(sentences[i].split()) for i in member_idx]))
        category = "junk" if avg_words < config.junk_avg_word_threshold else dominant

        center = norm_embeddings[member_idx].mean(axis=0)
        center = center / np.linalg.norm(center)
        sims = norm_embeddings[member_idx] @ center
        order = np.argsort(-sims)

        results.append(ClusterMembers(
            cluster_id=int(cid),
            category=category,
            sentiment_breakdown=dict(Counter(member_sentiments)),
            avg_words=avg_words,
            member_indices_ranked=member_idx[order].tolist(),
            similarities=sims[order].tolist(),
        ))
    return results
