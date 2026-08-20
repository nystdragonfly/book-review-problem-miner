"""
Centralized pipeline configuration.

Every model name, algorithm parameter, and threshold that was previously
a magic number scattered across scratch/*.py scripts lives here instead.
Values match what was actually validated in scratch/07-10 — this isn't a
fresh guess, it's the settled-on numbers with their reasoning written down
once instead of repeated (or silently drifting) across scripts.

Deliberately does NOT include "which book/dataset to run on" — that's a
per-run target, not a pipeline setting, and depends on the data-source
abstraction (still being designed). Keeping the two separate means this
config is stable regardless of how that abstraction shakes out.
"""
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PipelineConfig:
    # --- embedding ---
    embedding_model: str = "all-mpnet-base-v2"
    embedding_device: str = "cuda"

    # --- cleaning ---
    # Reviews at or under this length get dropped before sentence-splitting.
    # Validated against real output: legitimate short opinions ("I love
    # graphic novels.") survive at this threshold; pure junk mostly doesn't
    # reach the review stage at all (it gets caught by the per-cluster junk
    # check below, at the sentence level, instead).
    min_review_chars: int = 50

    # --- dimensionality reduction (UMAP), before clustering ---
    # Plain HDBSCAN on raw embeddings failed (curse of dimensionality —
    # see devlog 2026-08-17). Reducing to 5 dims first is the fix.
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    # Actual n_neighbors passed to UMAP is min(this, n_sentences - 1) --
    # small datasets (e.g. the synthetic one, ~400 sentences) need a
    # smaller neighborhood than the UMAP default of 15.
    umap_max_neighbors: int = 15

    # --- clustering (HDBSCAN) ---
    # 10 empirically separated a real 29-sentence artwork-complaint
    # cluster from a much larger general-negativity cluster on Watchmen;
    # confirmed to work reasonably on the much smaller synthetic dataset
    # too (13 clusters from ~424 sentences). Revisit if a new dataset's
    # scale makes it clearly wrong rather than assuming it's universal.
    hdbscan_min_cluster_size: int = 10
    hdbscan_metric: str = "euclidean"

    # --- categorization ---
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_batch_size: int = 64
    # Clusters averaging under this many words/sentence are "junk" (bare
    # interjections/numbers). Empirically: junk clusters averaged 1.3-2.4
    # words; the next-lowest legitimate cluster averaged 5.0 -- clean gap.
    junk_avg_word_threshold: float = 4.0

    # Sentences the small sentiment classifier calls "negative" AND that
    # contain a negation marker get a second opinion from the local LLM.
    # Only acts on negative->positive disagreements (validated as
    # near-100% reliable on manual spot-check, see devlog 2026-08-20) --
    # negative->neutral disagreements are NOT auto-applied, since that
    # bucket was a genuine mixed bag (real corrections mixed with
    # regressions on terse dismissals like "Not for me.").
    enable_negation_recheck: bool = True

    # --- labeling ---
    # Local Ollama, not a paid API -- keeps the whole pipeline local/free,
    # consistent with every other model used. Same model already running
    # for the user's own local-AI setup, so no extra memory overhead.
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "mistral-nemo:12b"
    ollama_timeout_seconds: int = 120

    # --- caching ---
    cache_dir: Path = REPO_ROOT / ".cache"

    # --- reproducibility ---
    random_state: int = 42


DEFAULT_CONFIG = PipelineConfig()
