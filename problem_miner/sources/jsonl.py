"""
Plain (non-gzipped) JSONL review files, one JSON object per line. Used
for the synthetic Aria-7 dataset today, but the format itself isn't
synthetic-specific — any future JSONL-shaped source, real or synthetic,
can reuse this as-is provided it already uses the canonical field names
(book_id, review_id, rating, review_text). A source with different field
names would need its own class that maps into RawReview, same idea as
GoodreadsSource but for a different schema.

Filters by book_id even though today's synthetic file already only
contains one book's reviews — treating "the file is pre-scoped to one
book" as an assumption to verify, not something to silently rely on.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from .base import RawReview


@dataclass
class JsonlSource:
    reviews_file: Path
    name: str = "jsonl"

    def load(self, book_id: str) -> list[RawReview]:
        reviews = []
        with open(self.reviews_file, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d["book_id"] == book_id:
                    reviews.append(RawReview(
                        book_id=d["book_id"],
                        review_id=d["review_id"],
                        rating=d["rating"],
                        review_text=d["review_text"],
                    ))
        return reviews
