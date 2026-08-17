"""
Scratch script — NOT pipeline code. First clustering pass.

1. Embed all Watchmen complaint sentences (cached — see common.py).
2. L2-normalize so Euclidean distance in k-means behaves like cosine
   similarity (what we actually care about for "meaning is similar").
3. Compute inertia + silhouette score across a range of k, as a rough
   guide — but the real decision comes from reading the clusters below,
   not from these numbers alone.
4. Run k-means at a couple of candidate k values and print, for each
   cluster: its size and the sentences closest to its centroid (i.e. the
   most "representative" examples of whatever theme it found).
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from common import load_cleaned_english_reviews, sentences_with_metadata, embed_sentences_cached, WATCHMEN_ID

reviews = load_cleaned_english_reviews(WATCHMEN_ID)
items = sentences_with_metadata(reviews)
sentences = [item["sentence"] for item in items]

raw_embeddings = embed_sentences_cached(sentences, cache_name="watchmen_sentences")
embeddings = normalize(raw_embeddings)  # L2 normalize -> Euclidean distance ~ cosine similarity
print(f"{len(sentences)} sentences, embeddings shape {embeddings.shape}\n")

# --- Step 1: elbow + silhouette across a range of k, as a rough guide ---
print("=" * 80)
print("k-selection metrics (guide only, not the final decision)")
print("=" * 80)
print(f"{'k':>4} {'inertia':>12} {'silhouette':>12}")
candidate_ks = list(range(4, 25, 2))
for k in candidate_ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(embeddings)
    # silhouette on the full 5.5k points is somewhat slow; subsample for speed
    sil = silhouette_score(embeddings, labels, sample_size=2000, random_state=42)
    print(f"{k:>4} {km.inertia_:>12.1f} {sil:>12.3f}")
print()

# --- Step 2: run at a couple of specific k values, print readable clusters ---
def show_clusters(k: int, n_examples: int = 5):
    print("=" * 80)
    print(f"k = {k}")
    print("=" * 80)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(embeddings)
    centers = normalize(km.cluster_centers_)  # re-normalize centers for cosine comparison

    for cluster_id in range(k):
        member_idx = np.where(labels == cluster_id)[0]
        # rank members by cosine similarity to their own cluster's center
        sims = embeddings[member_idx] @ centers[cluster_id]
        order = member_idx[np.argsort(-sims)]
        print(f"\n--- cluster {cluster_id}  (n={len(member_idx)}) ---")
        for idx in order[:n_examples]:
            print(f"  - {sentences[idx]}")


for k in [8, 14]:
    show_clusters(k)
