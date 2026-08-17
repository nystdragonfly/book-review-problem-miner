"""
Scratch script — NOT pipeline code. First embeddings sanity check.

Before spending effort on full k-means clustering, confirm the embeddings
actually capture *meaning* rather than just literal word overlap. We do
this by picking a couple of "query" sentences and finding their nearest
neighbors (by cosine similarity) among all the Watchmen complaint
sentences — including a query that's NOT verbatim in the dataset, to
check the model generalizes rather than just matching shared words.
"""
from sentence_transformers import SentenceTransformer

from common import load_cleaned_english_reviews, sentences_with_metadata, WATCHMEN_ID

reviews = load_cleaned_english_reviews(WATCHMEN_ID)
items = sentences_with_metadata(reviews)
sentences = [item["sentence"] for item in items]
print(f"{len(reviews)} cleaned reviews -> {len(sentences)} sentences\n")

print("Loading all-mpnet-base-v2 onto GPU...")
model = SentenceTransformer("all-mpnet-base-v2", device="cuda")

print("Encoding all sentences...")
embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=True)
print(f"embeddings shape: {tuple(embeddings.shape)}\n")

# --- Sanity check: nearest-neighbor search for a couple of hand-picked queries ---
queries = [
    "I found it a bit boring.",          # verbatim sentence FROM the dataset
    "the artwork was hard to follow",    # NOT in the dataset verbatim — tests generalization
]

query_embeddings = model.encode(queries, convert_to_tensor=True)
similarities = model.similarity(query_embeddings, embeddings)  # cosine sim, shape (n_queries, n_sentences)

for qi, query in enumerate(queries):
    print("=" * 80)
    print(f"QUERY: {query!r}")
    print("-" * 80)
    scores = similarities[qi]
    top_k = scores.topk(6)
    for score, idx in zip(top_k.values.tolist(), top_k.indices.tolist()):
        print(f"  [{score:.3f}] {sentences[idx]}")
    print()
