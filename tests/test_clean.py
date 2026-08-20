from problem_miner.clean import clean_reviews, is_english, normalize_whitespace
from problem_miner.config import PipelineConfig
from problem_miner.sources.base import RawReview


def test_normalize_whitespace_collapses_and_strips():
    assert normalize_whitespace("  a  b\n\n c \t d  ") == "a b c d"


def test_normalize_whitespace_handles_literal_newline_artifacts():
    # Real Goodreads data has stray " \n " sequences mid-sentence -- see
    # devlog 2026-08-17. This is the exact pattern that motivated the fix.
    assert normalize_whitespace("ran for about one \n year in 12 issues") == \
        "ran for about one year in 12 issues"


def test_is_english_true_for_clearly_english_text():
    assert is_english("This book was a genuinely great read from start to finish.") is True


def test_is_english_false_for_clearly_non_english_text():
    assert is_english("Este libro fue una lectura genuinamente excelente de principio a fin.") is False


def make_review(book_id="b1", review_id="r1", rating=3, text="x" * 60) -> RawReview:
    return RawReview(book_id=book_id, review_id=review_id, rating=rating, review_text=text)


def test_clean_reviews_drops_5_star():
    reviews = [make_review(rating=5)]
    assert clean_reviews(reviews) == []


def test_clean_reviews_drops_unrated_zero():
    # rating 0 means "shelved but not rated" on Goodreads, not a 1-star
    # complaint -- see CLAUDE.md.
    reviews = [make_review(rating=0)]
    assert clean_reviews(reviews) == []


def test_clean_reviews_drops_short_text():
    short_config = PipelineConfig(min_review_chars=50)
    reviews = [make_review(rating=3, text="too short")]
    assert clean_reviews(reviews, short_config) == []


def test_clean_reviews_keeps_valid_review_and_normalizes_whitespace():
    text = (
        "This   was  a genuinely   frustrating book to get through, "
        "and I really did not enjoy the pacing or the ending at all."
    )
    reviews = [make_review(rating=2, text=text)]
    result = clean_reviews(reviews)
    assert len(result) == 1
    assert "  " not in result[0].review_text  # no double spaces survive
    assert result[0].rating == 2
    assert result[0].review_id == "r1"
