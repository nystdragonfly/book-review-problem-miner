"""
Scratch script — NOT pipeline code. Diagnostic: we saw clear artwork
complaints in the nearest-neighbor check (03) but no artwork-complaint
cluster in the k-means output (04). Where did those specific sentences
actually get assigned?
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from common import load_cleaned_english_reviews, sentences_with_metadata, embed_sentences_cached, WATCHMEN_ID

reviews = load_cleaned_english_reviews(WATCHMEN_ID)
items = sentences_with_metadata(reviews)
sentences = [item["sentence"] for item in items]

raw_embeddings = embed_sentences_cached(sentences, cache_name="watchmen_sentences")
embeddings = normalize(raw_embeddings)

# The specific artwork complaints that surfaced in script 03's nearest-neighbor check
known_artwork_complaints = [
    "I found the art to distractingly poor.",
    "Didn't like the artwork.",
    "The artwork was a bit 'flat' for my liking.",
    "The juxtaposition of several storylines left me confused, and I didn't get as much out of the artwork as I think most devotees would.",
    "In addition to finding the story only slightly above average, I found the style of the artist to be annoying.",
    "It was difficult to read.",
]

# Find their indices in the sentence list (exact match)
indices = {}
for target in known_artwork_complaints:
    for i, s in enumerate(sentences):
        if s == target:
            indices[target] = i
            break

print(f"found {len(indices)}/{len(known_artwork_complaints)} target sentences in the dataset\n")

for k in [8, 14]:
    print("=" * 80)
    print(f"k = {k}")
    print("=" * 80)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(embeddings)
    centers = normalize(km.cluster_centers_)

    for target, idx in indices.items():
        cluster_id = labels[idx]
        cluster_size = (labels == cluster_id).sum()
        print(f"\n'{target[:70]}...'")
        print(f"  -> cluster {cluster_id} (size {cluster_size})")
        # show what ELSE is in that cluster, closest to centroid, for context
        member_idx = np.where(labels == cluster_id)[0]
        sims = embeddings[member_idx] @ centers[cluster_id]
        order = member_idx[np.argsort(-sims)]
        print(f"  cluster {cluster_id}'s top members (what it's actually 'about'):")
        for i in order[:4]:
            print(f"    - {sentences[i]}")
    print()

# Also: how many artwork-related sentences (rough keyword proxy) even exist?
art_keywords = ["artwork", "art style", "illustration", "drawings", "the art "]
art_related = [s for s in sentences if any(kw in s.lower() for kw in art_keywords)]
print("=" * 80)
print(f"{len(art_related)} / {len(sentences)} sentences mention art/artwork/illustration at all (rough keyword count)")
