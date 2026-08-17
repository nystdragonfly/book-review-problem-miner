"""
Scratch script — NOT pipeline code. UMAP (dimensionality reduction) +
HDBSCAN, the standard BERTopic-style recipe, since raw HDBSCAN on
768-dim embeddings failed (curse of dimensionality — see script 06).
"""
import numpy as np
import umap
from sklearn.cluster import HDBSCAN

from common import load_cleaned_english_reviews, sentences_with_metadata, embed_sentences_cached, WATCHMEN_ID

reviews = load_cleaned_english_reviews(WATCHMEN_ID)
items = sentences_with_metadata(reviews)
sentences = [item["sentence"] for item in items]

raw_embeddings = embed_sentences_cached(sentences, cache_name="watchmen_sentences")

known_artwork_complaints = [
    "I found the art to distractingly poor.",
    "Didn't like the artwork.",
    "The artwork was a bit 'flat' for my liking.",
]
target_indices = [sentences.index(s) for s in known_artwork_complaints]

print("Reducing 768-dim embeddings to 5-dim with UMAP (metric='cosine')...")
reducer = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
reduced = reducer.fit_transform(raw_embeddings)
print(f"reduced shape: {reduced.shape}\n")

for min_cluster_size in [10, 15, 25, 40]:
    print("=" * 80)
    print(f"min_cluster_size = {min_cluster_size}")
    print("=" * 80)
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = hdb.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    sizes = sorted([(labels == c).sum() for c in set(labels) if c != -1], reverse=True)
    print(f"{n_clusters} clusters, {n_noise}/{len(labels)} noise ({100*n_noise/len(labels):.1f}%)")
    print(f"cluster sizes: {sizes}")

    for s, idx in zip(known_artwork_complaints, target_indices):
        cid = labels[idx]
        print(f"  '{s[:60]}...' -> cluster {cid}" + (" (NOISE)" if cid == -1 else f" (size {(labels==cid).sum()})"))
    print()

# Full dump at whichever setting looks most balanced (adjust after seeing above)
# min_cluster_size=10 confirmed to separate art PRAISE from art COMPLAINTS as
# distinct clusters (15 only separated "art" as one mixed-sentiment topic) —
# see scratch/06/07 discussion. Finer granularity wins for our use case.
CHOSEN_MIN_CLUSTER_SIZE = 10
print("=" * 80)
print(f"Full cluster dump at min_cluster_size={CHOSEN_MIN_CLUSTER_SIZE}")
print("=" * 80)
hdb = HDBSCAN(min_cluster_size=CHOSEN_MIN_CLUSTER_SIZE, metric="euclidean")
labels = hdb.fit_predict(reduced)
cluster_ids = sorted(set(labels) - {-1})
print(f"{len(cluster_ids)} clusters, {(labels==-1).sum()} noise points\n")

norm_raw = raw_embeddings / np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
for cid in cluster_ids:
    member_idx = np.where(labels == cid)[0]
    # rank by similarity in the ORIGINAL embedding space (more meaningful than UMAP space for reading text similarity)
    center = norm_raw[member_idx].mean(axis=0)
    center = center / np.linalg.norm(center)
    sims = norm_raw[member_idx] @ center
    order = member_idx[np.argsort(-sims)]
    print(f"--- cluster {cid} (n={len(member_idx)}) ---")
    for i in order[:5]:
        print(f"  - {sentences[i]}")
    print()
