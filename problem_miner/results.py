"""
Structured pipeline output.

Design call: the pipeline does NOT pick a "top N" or rank clusters by
importance. It computes everything and writes it all out faithfully as
data — every cluster, every member sentence, full sentiment breakdowns.
Deciding what's worth showing (top N complaints, filtered by category,
sorted by size vs. negativity, whatever) is a presentation-layer concern
for whatever actually consumes this file (a CLI viewer today, possibly a
real interface later) — not something to bake into the pipeline itself.

That also sidesteps a real problem with picking a ranking rule too early:
the "right" ranking depends on who's looking at it and why (a reader
wants different information than a publisher doing quality control), and
the pipeline has no way to know that. Keep it dumb and complete; let the
consumer be opinionated.
"""
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class SentenceRecord:
    text: str
    review_id: str
    rating: int
    sentiment: str  # "positive" / "negative" / "neutral" — this sentence's own classification
    similarity_to_cluster_center: float  # lets a consumer re-rank within a cluster if it wants to


@dataclass
class ClusterResult:
    cluster_id: int
    category: str  # "negative" | "positive" | "neutral" | "junk"
    size: int
    avg_words_per_sentence: float
    sentiment_breakdown: dict[str, int]  # e.g. {"negative": 35, "neutral": 9, "positive": 9}
    title: str
    summary: str
    sentences: list[SentenceRecord]  # ALL members, not just a curated top-5 -- see module docstring


@dataclass
class PipelineResults:
    book_id: str
    source_name: str  # which data source this ran against, e.g. "synthetic-aria7", "watchmen"
    generated_at: str  # ISO 8601, set automatically -- don't pass this in
    total_raw_reviews: int
    cleaned_reviews: int
    total_sentences: int
    noise_sentence_count: int  # sentences HDBSCAN didn't assign to any cluster
    clusters: list[ClusterResult]

    @classmethod
    def create(cls, *, book_id: str, source_name: str, total_raw_reviews: int,
               cleaned_reviews: int, total_sentences: int, noise_sentence_count: int,
               clusters: list[ClusterResult]) -> "PipelineResults":
        return cls(
            book_id=book_id, source_name=source_name,
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_raw_reviews=total_raw_reviews, cleaned_reviews=cleaned_reviews,
            total_sentences=total_sentences, noise_sentence_count=noise_sentence_count,
            clusters=clusters,
        )

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str | Path) -> "PipelineResults":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        clusters = [
            ClusterResult(**{**c, "sentences": [SentenceRecord(**s) for s in c["sentences"]]})
            for c in data["clusters"]
        ]
        return cls(**{**data, "clusters": clusters})
