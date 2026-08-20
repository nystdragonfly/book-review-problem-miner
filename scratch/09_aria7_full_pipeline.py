"""
Scratch script — NOT pipeline code. Run the validated pipeline (clean ->
split -> embed -> UMAP+HDBSCAN cluster -> sentiment/junk categorize) end
to end on the synthetic Aria-7 dataset, mirroring 07+08 but pointed at
the new, cleanly-licensed data instead of Watchmen.

Same approach as before, applied fresh — not re-litigating k-means vs
HDBSCAN (already settled, see devlog.md), just checking whether it holds
up on a much smaller, synthetic dataset.
"""
from collections import Counter

import numpy as np
import umap
from sklearn.cluster import HDBSCAN
from transformers import pipeline

from common import (
    load_reviews_from_jsonl, clean_reviews, sentences_with_metadata,
    embed_sentences_cached, ARIA7_REVIEWS_FILE,
)

raw = load_reviews_from_jsonl(ARIA7_REVIEWS_FILE)
reviews = clean_reviews(raw)
items = sentences_with_metadata(reviews)
sentences = [item["sentence"] for item in items]
print(f"{len(raw)} raw -> {len(reviews)} cleaned reviews -> {len(sentences)} sentences\n")

raw_embeddings = embed_sentences_cached(sentences, cache_name="aria7_sentences")

print("Reducing dimensionality with UMAP...")
# n_neighbors capped below the default 15 since we have far fewer sentences
# than the Watchmen run (few hundred vs several thousand) -- UMAP needs
# n_neighbors < n_samples, and a smaller neighborhood makes more sense at
# this scale anyway.
n_neighbors = min(15, len(sentences) - 1)
reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
reduced = reducer.fit_transform(raw_embeddings)

MIN_CLUSTER_SIZE = 10
print(f"Clustering with HDBSCAN (min_cluster_size={MIN_CLUSTER_SIZE})...")
hdb = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric="euclidean")
labels = hdb.fit_predict(reduced)
cluster_ids = sorted(set(labels) - {-1})
print(f"{len(cluster_ids)} clusters, {(labels == -1).sum()} noise points out of {len(sentences)}\n")

print("Loading sentiment classifier onto GPU...")
sentiment_clf = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0,
    truncation=True,
)
print(f"Classifying sentiment for all {len(sentences)} sentences...")
sentiment_results = sentiment_clf(sentences, batch_size=64)
sentiment_labels = [r["label"] for r in sentiment_results]
print(f"overall distribution: {Counter(sentiment_labels)}\n")

JUNK_AVG_WORD_THRESHOLD = 4
norm_raw = raw_embeddings / np.linalg.norm(raw_embeddings, axis=1, keepdims=True)

cluster_summaries = []
for cid in cluster_ids:
    member_idx = np.where(labels == cid)[0]
    member_sentiments = [sentiment_labels[i] for i in member_idx]
    sentiment_counts = Counter(member_sentiments)
    dominant, dominant_count = sentiment_counts.most_common(1)[0]
    dominant_pct = 100 * dominant_count / len(member_idx)

    avg_words = np.mean([len(sentences[i].split()) for i in member_idx])
    is_junk = avg_words < JUNK_AVG_WORD_THRESHOLD
    category = "junk" if is_junk else dominant

    center = norm_raw[member_idx].mean(axis=0)
    center = center / np.linalg.norm(center)
    sims = norm_raw[member_idx] @ center
    order = member_idx[np.argsort(-sims)]

    cluster_summaries.append({
        "cluster_id": cid, "size": len(member_idx), "category": category,
        "avg_words": avg_words, "dominant_sentiment": dominant, "dominant_pct": dominant_pct,
        "sentiment_counts": sentiment_counts,
        "top_examples": [sentences[i] for i in order[:5]],
    })

CATEGORY_ORDER = {"negative": 0, "positive": 1, "neutral": 2, "junk": 3}
cluster_summaries.sort(key=lambda c: (CATEGORY_ORDER[c["category"]], -c["size"]))

TAG_LABELS = {
    "negative": "NEGATIVE (complaint)", "positive": "POSITIVE (praise)",
    "neutral": "NEUTRAL (informative)", "junk": "JUNK (no real content)",
}

for c in cluster_summaries:
    tag = TAG_LABELS[c["category"]]
    print("=" * 80)
    print(f"cluster {c['cluster_id']}  n={c['size']}  avg_words={c['avg_words']:.1f}  -> {tag}")
    if c["category"] != "junk":
        print(f"  ({c['dominant_pct']:.0f}% {c['dominant_sentiment']})  sentiment breakdown: {dict(c['sentiment_counts'])}")
    print("-" * 80)
    for s in c["top_examples"]:
        print(f"  - {s}")
    print()

category_totals = Counter(c["category"] for c in cluster_summaries)
sentence_totals = Counter()
for c in cluster_summaries:
    sentence_totals[c["category"]] += c["size"]
print("=" * 80)
print(f"cluster counts by category: {dict(category_totals)}")
print(f"sentence counts by category: {dict(sentence_totals)}")
