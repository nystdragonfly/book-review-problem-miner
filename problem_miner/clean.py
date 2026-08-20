"""
Cleaning: drop 5-star/unrated reviews, drop short reviews, normalize
whitespace, drop non-English reviews. Ported from the validated
scratch/common.py logic (see CLAUDE.md "Where things stand"), now
operating on RawReview objects instead of loose dicts.
"""
import re

from langdetect import LangDetectException, detect

from .config import DEFAULT_CONFIG, PipelineConfig
from .sources.base import RawReview


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        # detect() throws on text with no linguistic signal (e.g. just
        # emoji or numbers) -- treat as "can't tell, drop it".
        return False


def clean_reviews(reviews: list[RawReview], config: PipelineConfig = DEFAULT_CONFIG) -> list[RawReview]:
    usable = [
        r for r in reviews
        if r.rating not in (0, 5) and len(r.review_text) > config.min_review_chars
    ]
    cleaned = []
    for r in usable:
        text = normalize_whitespace(r.review_text)
        if is_english(text):
            cleaned.append(RawReview(
                book_id=r.book_id, review_id=r.review_id, rating=r.rating, review_text=text,
            ))
    return cleaned
