"""
Scratch script — NOT pipeline code. Re-run the validated UMAP+HDBSCAN
clustering (see 07), then label each cluster positive/negative/neutral
by classifying each member sentence's own text with a dedicated 3-class
sentiment model (NOT the source review's star rating — see the trade-off
discussion: rating-average is noisy at the sentence level, this isn't).

"neutral" doubles as our "informative" bucket (content notes like "this
is intended for mature readers" should score neutral, not positive/negative).
"""
from collections import Counter

import numpy as np
import umap
from sklearn.cluster import HDBSCAN
from transformers import pipeline

from common import load_cleaned_english_reviews, sentences_with_metadata, embed_sentences_cached, WATCHMEN_ID

reviews = load_cleaned_english_reviews(WATCHMEN_ID)
items = sentences_with_metadata(reviews)
sentences = [item["sentence"] for item in items]

raw_embeddings = embed_sentences_cached(sentences, cache_name="watchmen_sentences")

print("Reducing dimensionality with UMAP...")
reducer = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
reduced = reducer.fit_transform(raw_embeddings)

print("Clustering with HDBSCAN (min_cluster_size=10)...")
hdb = HDBSCAN(min_cluster_size=10, metric="euclidean")
labels = hdb.fit_predict(reduced)
cluster_ids = sorted(set(labels) - {-1})
print(f"{len(cluster_ids)} clusters, {(labels == -1).sum()} noise points\n")

print("Loading sentiment classifier (cardiffnlp/twitter-roberta-base-sentiment-latest) onto GPU...")
sentiment_clf = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0,  # GPU
    truncation=True,
)

print(f"Classifying sentiment for all {len(sentences)} sentences...")
sentiment_results = sentiment_clf(sentences, batch_size=64)
sentiment_labels = [r["label"] for r in sentiment_results]  # 'positive' / 'negative' / 'neutral'
print(f"overall distribution: {Counter(sentiment_labels)}\n")

# --- Label each cluster by its dominant sentiment, print examples ---
norm_raw = raw_embeddings / np.linalg.norm(raw_embeddings, axis=1, keepdims=True)

# Clusters averaging under this many words per sentence are "junk" —
# bare interjections/numbers with no real content ("No.", "4.", "Really?").
# Empirically: known junk clusters average 1.3-2.4 words; the next-lowest
# legitimate cluster averages 5.0 words ("So I did.", "I'm glad I did.") —
# clean gap, so this threshold isn't fine-tuned to force a specific split.
JUNK_AVG_WORD_THRESHOLD = 4

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
        "cluster_id": cid,
        "size": len(member_idx),
        "category": category,
        "avg_words": avg_words,
        "dominant_sentiment": dominant,
        "dominant_pct": dominant_pct,
        "sentiment_counts": sentiment_counts,
        "top_examples": [sentences[i] for i in order[:5]],
    })

# Sort by category so the four groups are easy to scan block-by-block,
# biggest-first within each category.
CATEGORY_ORDER = {"negative": 0, "positive": 1, "neutral": 2, "junk": 3}
cluster_summaries.sort(key=lambda c: (CATEGORY_ORDER[c["category"]], -c["size"]))

TAG_LABELS = {
    "negative": "NEGATIVE (complaint)",
    "positive": "POSITIVE (praise)",
    "neutral": "NEUTRAL (informative)",
    "junk": "JUNK (no real content)",
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
