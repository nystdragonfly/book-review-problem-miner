"""
The canonical shape every data source normalizes into, and the interface
every source implements.

The rest of the pipeline only ever sees RawReview objects — it doesn't
know or care whether a review came from a gzip'd multi-book firehose, a
plain single-book JSONL file, or (someday) something else entirely. File
format and field-naming differences are the source's problem to solve
internally, not something that leaks into clean.py/split.py/etc.
"""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RawReview:
    book_id: str
    review_id: str
    rating: int
    review_text: str


class ReviewSource(Protocol):
    """Every concrete source (GoodreadsSource, JsonlSource, and whatever
    gets added later) implements this. `name` identifies the source in
    PipelineResults.source_name for provenance."""
    name: str

    def load(self, book_id: str) -> list[RawReview]:
        """Return every raw review record for the given book_id, mapped
        into the canonical RawReview shape regardless of the source's
        native format/field names."""
        ...
