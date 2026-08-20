from problem_miner.sources.base import RawReview
from problem_miner.split import split_into_sentences


def test_split_into_sentences_basic():
    review = RawReview(
        book_id="b1", review_id="r1", rating=4,
        review_text="I loved this book. The pacing was a bit slow though.",
    )
    sentences = split_into_sentences([review])
    assert len(sentences) == 2
    assert sentences[0].text == "I loved this book."
    assert sentences[1].text == "The pacing was a bit slow though."


def test_split_into_sentences_carries_metadata():
    review = RawReview(book_id="b1", review_id="r42", rating=2, review_text="Just okay.")
    sentences = split_into_sentences([review])
    assert sentences[0].review_id == "r42"
    assert sentences[0].rating == 2


def test_split_into_sentences_handles_multiple_reviews():
    reviews = [
        RawReview(book_id="b1", review_id="r1", rating=3, review_text="One sentence here."),
        RawReview(book_id="b1", review_id="r2", rating=4, review_text="First. Second."),
    ]
    sentences = split_into_sentences(reviews)
    assert len(sentences) == 3
    assert [s.review_id for s in sentences] == ["r1", "r2", "r2"]
