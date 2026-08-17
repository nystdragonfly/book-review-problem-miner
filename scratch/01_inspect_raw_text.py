"""
Scratch script — NOT pipeline code. Just looking at raw Watchmen review text
to see what kinds of "noise" actually show up, before deciding how to clean it.
"""
import gzip
import json
from pathlib import Path

WATCHMEN_ID = "472331"

# Resolve relative to this file, not the process's working directory —
# makes the script runnable from anywhere (terminal, PyCharm's default
# per-script working dir, etc.) without needing `cd` first.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

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

print(f"{len(usable)} usable reviews\n")

# Look at a spread: a few short ones, a few long ones, a few in between,
# sorted by length so we see variety rather than just whatever's first.
usable.sort(key=lambda r: len(r["review_text"]))
sample_indices = [0, 1, 2, len(usable)//4, len(usable)//2, 3*len(usable)//4, -3, -2, -1]

for i in sample_indices:
    r = usable[i]
    print("=" * 80)
    print(f"rating={r['rating']}  len={len(r['review_text'])} chars")
    print("-" * 80)
    print(r["review_text"][:600])
    print()
