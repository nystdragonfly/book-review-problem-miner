import gzip
import json

from problem_miner.sources.base import RawReview
from problem_miner.sources.goodreads import GoodreadsSource
from problem_miner.sources.jsonl import JsonlSource

RECORDS = [
    {"book_id": "book-a", "review_id": "r1", "rating": 4, "review_text": "Great read."},
    {"book_id": "book-a", "review_id": "r2", "rating": 2, "review_text": "Not for me."},
    {"book_id": "book-b", "review_id": "r3", "rating": 5, "review_text": "Different book entirely."},
]


def test_goodreads_source_filters_by_book_id_from_multi_book_file(tmp_path):
    path = tmp_path / "reviews.json.gz"
    with gzip.open(path, "wt") as f:
        for r in RECORDS:
            f.write(json.dumps(r) + "\n")

    source = GoodreadsSource(reviews_file=path)
    result = source.load("book-a")

    assert len(result) == 2
    assert all(r.book_id == "book-a" for r in result)
    assert result[0] == RawReview(book_id="book-a", review_id="r1", rating=4, review_text="Great read.")


def test_goodreads_source_returns_empty_for_unknown_book_id(tmp_path):
    path = tmp_path / "reviews.json.gz"
    with gzip.open(path, "wt") as f:
        for r in RECORDS:
            f.write(json.dumps(r) + "\n")

    source = GoodreadsSource(reviews_file=path)
    assert source.load("nonexistent-book") == []


def test_jsonl_source_filters_by_book_id(tmp_path):
    path = tmp_path / "reviews.jsonl"
    with open(path, "w") as f:
        for r in RECORDS:
            f.write(json.dumps(r) + "\n")

    source = JsonlSource(reviews_file=path)
    result = source.load("book-b")

    assert len(result) == 1
    assert result[0].review_id == "r3"


def test_jsonl_source_skips_blank_lines(tmp_path):
    path = tmp_path / "reviews.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps(RECORDS[0]) + "\n")
        f.write("\n")  # blank line, e.g. trailing newline in the file
        f.write(json.dumps(RECORDS[1]) + "\n")

    source = JsonlSource(reviews_file=path)
    result = source.load("book-a")
    assert len(result) == 2


def test_default_source_name_differs_between_source_types(tmp_path):
    path = tmp_path / "reviews.jsonl"
    path.write_text("")
    assert GoodreadsSource(reviews_file=path).name == "goodreads"
    assert JsonlSource(reviews_file=path).name == "jsonl"
