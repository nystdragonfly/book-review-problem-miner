"""
Scratch script — NOT pipeline code. Cluster labeling (pipeline step 6):
for each cluster, generate a title + one-sentence summary via a local
Ollama server. (An earlier version also tried TF-IDF keyword tags
alongside this — dropped after comparing both: the keywords were
inconsistent value, often redundant with the LLM title, sometimes just
noise. LLM title + summary alone was judged sufficient.)
"""
from collections import Counter

import numpy as np
import requests
import umap
from sklearn.cluster import HDBSCAN
from transformers import pipeline

from common import (
    load_reviews_from_jsonl, clean_reviews, sentences_with_metadata,
    embed_sentences_cached, ARIA7_REVIEWS_FILE,
)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral-nemo:12b"

BOOK_CONTEXT = (
    "ARIA-7, Book One is a dungeon-core / progression-fantasy novel about "
    "an AI (ARIA-7) that becomes a sapient dungeon core in a fantasy world "
    "after her creator, Thomas, dies. Thomas's consciousness stays present "
    "as a companion she can talk to but not touch. The book is written in "
    "an unusually introspective, analytical style rather than typical "
    "stat-screen/loot LitRPG conventions."
)

# --- rebuild the same cluster assignment as script 09 (fast, cached embeddings) ---
raw = load_reviews_from_jsonl(ARIA7_REVIEWS_FILE)
reviews = clean_reviews(raw)
items = sentences_with_metadata(reviews)
sentences = [item["sentence"] for item in items]

raw_embeddings = embed_sentences_cached(sentences, cache_name="aria7_sentences")
reducer = umap.UMAP(n_neighbors=min(15, len(sentences) - 1), n_components=5, min_dist=0.0, metric="cosine", random_state=42)
reduced = reducer.fit_transform(raw_embeddings)
hdb = HDBSCAN(min_cluster_size=10, metric="euclidean")
labels = hdb.fit_predict(reduced)
cluster_ids = sorted(set(labels) - {-1})
print(f"{len(cluster_ids)} clusters\n")

sentiment_clf = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0, truncation=True)
sentiment_labels = [r["label"] for r in sentiment_clf(sentences, batch_size=64)]

norm_raw = raw_embeddings / np.linalg.norm(raw_embeddings, axis=1, keepdims=True)


def get_llm_title_summary(example_sentences: list[str], category: str) -> tuple[str, str]:
    prompt = f"""{BOOK_CONTEXT}

Below are representative sentences from ONE cluster of reader review sentences, all grouped together because they discuss a similar theme. This cluster's overall category is: {category}.

Sentences:
{chr(10).join(f"- {s}" for s in example_sentences)}

Write a short title (5-8 words, naming the specific theme) and a one-sentence summary of what reviewers are saying about it.

Respond in EXACTLY this format, nothing else:
TITLE: <title>
SUMMARY: <one sentence>"""

    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
    }, timeout=120)
    resp.raise_for_status()
    text = resp.json()["response"].strip()

    # Models (this one included) like to add markdown bold around labels
    # even when told not to ("**TITLE:**" instead of "TITLE:") -- strip
    # markdown/whitespace before matching so parsing doesn't silently fail.
    def strip_markdown(s: str) -> str:
        return s.strip().strip("*").strip()

    title, summary = "(parse failed)", text
    for line in text.splitlines():
        cleaned = strip_markdown(line)
        if cleaned.upper().startswith("TITLE:"):
            title = strip_markdown(cleaned.split(":", 1)[1])
        elif cleaned.upper().startswith("SUMMARY:"):
            summary = strip_markdown(cleaned.split(":", 1)[1])
    return title, summary


JUNK_AVG_WORD_THRESHOLD = 4
results = []
for i, cid in enumerate(cluster_ids):
    member_idx = np.where(labels == cid)[0]
    member_sentiments = [sentiment_labels[j] for j in member_idx]
    dominant, dominant_count = Counter(member_sentiments).most_common(1)[0]
    avg_words = np.mean([len(sentences[j].split()) for j in member_idx])
    category = "junk" if avg_words < JUNK_AVG_WORD_THRESHOLD else dominant

    center = norm_raw[member_idx].mean(axis=0)
    center = center / np.linalg.norm(center)
    sims = norm_raw[member_idx] @ center
    order = member_idx[np.argsort(-sims)]
    top_examples = [sentences[j] for j in order[:5]]

    print(f"[{i+1}/{len(cluster_ids)}] cluster {cid} (n={len(member_idx)}, {category}) -- calling {OLLAMA_MODEL}...")
    title, summary = get_llm_title_summary(top_examples, category)

    results.append({
        "cluster_id": cid, "size": len(member_idx), "category": category,
        "title": title, "summary": summary, "top_examples": top_examples,
    })

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
for r in results:
    print(f"\n--- cluster {r['cluster_id']} (n={r['size']}, {r['category']}) ---")
    print(f"TITLE:   {r['title']}")
    print(f"SUMMARY: {r['summary']}")
    print(f"examples:")
    for s in r["top_examples"][:3]:
        print(f"  - {s}")
