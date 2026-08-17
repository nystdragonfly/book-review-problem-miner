"""
Scratch script — NOT pipeline code. Try HDBSCAN instead of k-means.

Unlike k-means, HDBSCAN doesn't force every point into a cluster — points
in low-density regions get labeled noise (-1) instead. We're specifically
checking whether this recovers a real "artwork complaints" cluster that
k-means couldn't find because it was too small/sparse relative to the
giant general-sentiment cloud around it.
"""
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize

from common import load_cleaned_english_reviews, sentences_with_metadata, embed_sentences_cached, WATCHMEN_ID

reviews = load_cleaned_english_reviews(WATCHMEN_ID)
items = sentences_with_metadata(reviews)
sentences = [item["sentence"] for item in items]

raw_embeddings = embed_sentences_cached(sentences, cache_name="watchmen_sentences")
embeddings = normalize(raw_embeddings)

known_artwork_complaints = [
    "I found the art to distractingly poor.",
    "Didn't like the artwork.",
    "The artwork was a bit 'flat' for my liking.",
]
target_indices = [sentences.index(s) for s in known_artwork_complaints]

for min_cluster_size in [10, 15, 25, 40]:
    print("=" * 80)
    print(f"min_cluster_size = {min_cluster_size}")
    print("=" * 80)
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = hdb.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"{n_clusters} clusters found, {n_noise}/{len(labels)} points labeled noise ({100*n_noise/len(labels):.1f}%)")

    # where did our known artwork complaints land?
    for s, idx in zip(known_artwork_complaints, target_indices):
        cid = labels[idx]
        print(f"  '{s[:60]}...' -> cluster {cid}" + (" (NOISE)" if cid == -1 else f" (size {(labels==cid).sum()})"))
    print()

# Use the setting that seems most promising to actually print out all clusters
print("=" * 80)
print("Full cluster dump at min_cluster_size=15")
print("=" * 80)
hdb = HDBSCAN(min_cluster_size=15, metric="euclidean")
labels = hdb.fit_predict(embeddings)
cluster_ids = sorted(set(labels) - {-1})
print(f"{len(cluster_ids)} clusters, {(labels==-1).sum()} noise points\n")

for cid in cluster_ids:
    member_idx = np.where(labels == cid)[0]
    center = embeddings[member_idx].mean(axis=0)
    center = center / np.linalg.norm(center)
    sims = embeddings[member_idx] @ center
    order = member_idx[np.argsort(-sims)]
    print(f"--- cluster {cid} (n={len(member_idx)}) ---")
    for i in order[:5]:
        print(f"  - {sentences[i]}")
    print()
