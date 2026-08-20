"""
Scratch helper module — shared by the numbered scratch/NN_*.py scripts so
each one doesn't re-implement loading + cleaning from scratch.

NOT pipeline code. Once the approach is validated end-to-end, this logic
gets rewritten properly (not just moved) into real pipeline code.
"""
import gzip
import json
import re
from pathlib import Path

import numpy as np
from langdetect import detect, LangDetectException
from nltk.tokenize import sent_tokenize

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COMICS_REVIEWS_FILE = DATA_DIR / "goodreads_reviews_comics_graphic.json.gz"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

WATCHMEN_ID = "472331"

SYNTHETIC_DATA_DIR = Path(__file__).resolve().parent.parent / "synthetic_data"
ARIA7_REVIEWS_FILE = SYNTHETIC_DATA_DIR / "aria7_reviews.jsonl"
ARIA7_BOOK_ID = "synthetic-aria7-b1"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def load_reviews_for_book(book_id: str) -> list[dict]:
    reviews = []
    with gzip.open(COMICS_REVIEWS_FILE, "rt") as f:
        for line in f:
            d = json.loads(line)
            if d["book_id"] == book_id:
                reviews.append(d)
    return reviews


def clean_reviews(reviews: list[dict], min_chars: int = 50) -> list[dict]:
    """Non-5-star, non-unrated, real-length, English-only, whitespace-normalized.
    Shared cleaning logic — takes any list of raw review dicts, regardless of
    which dataset they came from."""
    usable = [
        r for r in reviews
        if r["rating"] not in (0, 5) and len(r["review_text"]) > min_chars
    ]
    cleaned = []
    for r in usable:
        text = normalize_whitespace(r["review_text"])
        if is_english(text):
            cleaned.append({**r, "review_text": text})
    return cleaned


def load_cleaned_english_reviews(book_id: str, min_chars: int = 50) -> list[dict]:
    """Real (Goodreads/UCSD) dataset only — see load_reviews_from_jsonl +
    clean_reviews for the synthetic Aria-7 dataset."""
    reviews = load_reviews_for_book(book_id)
    return clean_reviews(reviews, min_chars)


def load_reviews_from_jsonl(path: Path) -> list[dict]:
    """Load the synthetic (plain, non-gzipped) JSONL review files."""
    reviews = []
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                reviews.append(json.loads(line))
    return reviews


def sentences_with_metadata(reviews: list[dict]) -> list[dict]:
    """Split each review into sentences, keeping a link back to its review
    (rating, review_id) so we can trace a sentence back to its source later."""
    out = []
    for r in reviews:
        for sentence in sent_tokenize(r["review_text"]):
            out.append({
                "sentence": sentence,
                "rating": r["rating"],
                "review_id": r["review_id"],
            })
    return out


def embed_sentences_cached(sentences: list[str], cache_name: str, model_name: str = "all-mpnet-base-v2"):
    """Encode sentences, caching the result to disk keyed by cache_name.
    Re-embedding 5000+ sentences every run while we tune clustering is
    pure waste — this skips it if a cached .npy already matches.
    NOTE: the cache does NOT check whether `sentences` changed since the
    file was written. Delete the .npy under scratch/.cache/ manually if
    the underlying data/cleaning logic changes.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_name}.npy"
    if cache_path.exists():
        print(f"[cache] loading embeddings from {cache_path}")
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer
    print(f"[cache] no cache found, encoding {len(sentences)} sentences with {model_name}...")
    model = SentenceTransformer(model_name, device="cuda")
    embeddings = model.encode(sentences, show_progress_bar=True)
    np.save(cache_path, embeddings)
    print(f"[cache] saved to {cache_path}")
    return embeddings
