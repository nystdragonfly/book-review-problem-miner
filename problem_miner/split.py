"""
Sentence splitting. Ported from scratch/common.py's sentences_with_metadata.

Sentence-level, not whole-review, is a deliberate choice — see the README
case study: a single review often mixes praise and complaint, and
whole-review embeddings blend the two into a mushy vector.
"""
from dataclasses import dataclass

from nltk.tokenize import sent_tokenize

from .sources.base import RawReview


@dataclass(frozen=True)
class Sentence:
    text: str
    review_id: str
    rating: int


def split_into_sentences(reviews: list[RawReview]) -> list[Sentence]:
    out = []
    for r in reviews:
        for sentence_text in sent_tokenize(r.review_text):
            out.append(Sentence(text=sentence_text, review_id=r.review_id, rating=r.rating))
    return out
