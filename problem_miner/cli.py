"""
Thin CLI entry point: `python -m problem_miner ...`

Deliberately source-agnostic -- --source picks which ReviewSource
implementation to use, everything else (book_id, output path, etc.) is
generic across sources. Doesn't hardcode any book-specific defaults; a
convenience wrapper for a *specific* book (e.g. a `run_aria7_demo.sh`)
can live outside this module if that's ever wanted, but the CLI itself
shouldn't know about any one book.
"""
import argparse
from pathlib import Path

from .config import DEFAULT_CONFIG
from .pipeline import run_pipeline
from .sources.goodreads import GoodreadsSource
from .sources.jsonl import JsonlSource

SOURCE_TYPES = {
    "jsonl": JsonlSource,
    "goodreads": GoodreadsSource,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="problem_miner",
        description="Mine book reviews for recurring complaint / praise / informative themes.",
    )
    parser.add_argument(
        "--source", required=True, choices=sorted(SOURCE_TYPES),
        help="Which data source implementation to use.",
    )
    parser.add_argument(
        "--reviews-file", required=True, type=Path,
        help="Path to the source's review data file.",
    )
    parser.add_argument(
        "--source-name", default=None,
        help="Label for this source in the output (defaults to the source type name).",
    )
    parser.add_argument("--book-id", required=True, help="book_id to filter reviews for.")
    parser.add_argument(
        "--book-context", default=None,
        help="Optional short description of the book, given to the labeling model as context.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/results.json"),
        help="Where to write the structured results JSON (default: output/results.json).",
    )
    parser.add_argument(
        "--n-examples", type=int, default=None,
        help="If set, keep only the top-K most representative sentences per cluster "
             "in the output instead of the full membership.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    source_cls = SOURCE_TYPES[args.source]
    kwargs = {"reviews_file": args.reviews_file}
    if args.source_name:
        kwargs["name"] = args.source_name
    source = source_cls(**kwargs)

    results = run_pipeline(
        source=source,
        book_id=args.book_id,
        book_context=args.book_context,
        config=DEFAULT_CONFIG,
        n_examples_per_cluster=args.n_examples,
    )
    results.save_json(args.output)

    print(f"\n{'=' * 70}")
    print(
        f"Done: {len(results.clusters)} clusters from {results.total_sentences} sentences "
        f"({results.cleaned_reviews}/{results.total_raw_reviews} reviews used, "
        f"{results.noise_sentence_count} sentences unclustered)"
    )
    print(f"Results written to {args.output}")
    print(f"{'=' * 70}")
    for c in sorted(results.clusters, key=lambda c: -c.size):
        print(f"  [{c.category:>9}] n={c.size:<4} {c.title}")


if __name__ == "__main__":
    main()
