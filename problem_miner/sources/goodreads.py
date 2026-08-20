"""
Real UCSD/Goodreads dataset — used for internal technical validation
only, never as a demo-facing data source. Its license is academic-use-
only/no-commercial-use, which is why the synthetic dataset exists at
all; see CLAUDE.md and README.md for the full reasoning. Keeping this
source in the codebase (rather than deleting it) is deliberate — it's
still the right tool for continued real-data validation work, just not
for anything shown or distributed.

The underlying file is a firehose of reviews for ~89,000 different
books in one gzip'd JSONL file, so loading means scanning the whole
file and filtering by book_id — there's no way to seek directly to one
book's reviews without an index this project doesn't build.
"""
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

from .base import RawReview


@dataclass
class GoodreadsSource:
    reviews_file: Path
    name: str = "goodreads"

    def load(self, book_id: str) -> list[RawReview]:
        reviews = []
        with gzip.open(self.reviews_file, "rt") as f:
            for line in f:
                d = json.loads(line)
                if d["book_id"] == book_id:
                    reviews.append(RawReview(
                        book_id=d["book_id"],
                        review_id=d["review_id"],
                        rating=d["rating"],
                        review_text=d["review_text"],
                    ))
        return reviews
