"""
Scratch script — NOT pipeline code. Cleaning pass + sentence splitting on
Watchmen reviews, so we can eyeball whether it actually helps before this
becomes real pipeline code.

Steps:
1. Normalize whitespace (collapse literal "\n" artifacts, extra spaces).
2. Filter out non-English reviews (langdetect).
3. Split into sentences (nltk).
"""
import gzip
import json
import re
from pathlib import Path

from langdetect import detect, LangDetectException
from nltk.tokenize import sent_tokenize

WATCHMEN_ID = "472331"

# Resolve relative to this file, not the process's working directory —
# makes the script runnable from anywhere (terminal, PyCharm's default
# per-script working dir, etc.) without needing `cd` first.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def normalize_whitespace(text: str) -> str:
    # Collapse any run of whitespace (including literal newlines) into a
    # single space. The raw text has stray " \n " sequences mid-sentence
    # that would otherwise look like paragraph breaks.
    return re.sub(r"\s+", " ", text).strip()


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        # detect() throws on text with no linguistic signal (e.g. just
        # emoji or numbers) — treat as "can't tell, drop it".
        return False


reviews = []
with gzip.open(DATA_DIR / "goodreads_reviews_comics_graphic.json.gz", "rt") as f:
    for line in f:
        d = json.loads(line)
        if d["book_id"] == WATCHMEN_ID:
            reviews.append(d)

usable = [
    r for r in reviews
    if r["rating"] not in (0, 5) and len(r["review_text"]) > 50
]
print(f"usable reviews before language filter: {len(usable)}")

cleaned = []
dropped_non_english = []
for r in usable:
    text = normalize_whitespace(r["review_text"])
    if is_english(text):
        cleaned.append({**r, "review_text": text})
    else:
        dropped_non_english.append(text)

print(f"usable reviews after language filter: {len(cleaned)}")
print(f"dropped as non-English: {len(dropped_non_english)}")
print()
print("--- sample of what got dropped (sanity check) ---")
for t in dropped_non_english[:5]:
    print(f"  {t[:100]}")
print()

# Now split a handful of cleaned reviews into sentences to see what
# snippet-level granularity actually looks like.
print("=" * 80)
print("SENTENCE SPLITTING EXAMPLES")
print("=" * 80)
for r in cleaned[len(cleaned)//2:len(cleaned)//2 + 3]:
    print("-" * 80)
    print(f"[full review, rating={r['rating']}]")
    print(r["review_text"][:300])
    print()
    sentences = sent_tokenize(r["review_text"])
    print(f"[{len(sentences)} sentences]")
    for s in sentences[:8]:
        print(f"  - {s}")
    print()
