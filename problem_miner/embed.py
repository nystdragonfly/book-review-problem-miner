"""
Sentence embedding, with an on-disk cache keyed by a hash of the actual
sentence content.

Fixes a known flaw in the scratch version (scratch/common.py's
embed_sentences_cached, see its docstring): that cache was keyed by a
caller-chosen name only and never checked whether the underlying
sentences had actually changed, so a stale cache could silently get
reused after cleaning/splitting logic changed elsewhere. Here, a
mismatched content hash is treated as a cache miss and embeddings are
recomputed automatically instead of requiring a manual .npy delete.
"""
import hashlib

import numpy as np

from .config import DEFAULT_CONFIG, PipelineConfig


def _sentence_hash(sentences: list[str]) -> str:
    joined = "\n".join(sentences).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


def embed_sentences(sentences: list[str], cache_key: str, config: PipelineConfig = DEFAULT_CONFIG) -> np.ndarray:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = config.cache_dir / f"{cache_key}.npz"
    expected_hash = _sentence_hash(sentences)

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["sentence_hash"].item()) == expected_hash:
            print(f"[cache] loading embeddings from {cache_path}")
            return cached["embeddings"]
        print(f"[cache] stale cache at {cache_path} (sentences changed since it was written) -- recomputing")

    from sentence_transformers import SentenceTransformer
    print(f"[embed] encoding {len(sentences)} sentences with {config.embedding_model}...")
    model = SentenceTransformer(config.embedding_model, device=config.embedding_device)
    embeddings = model.encode(sentences, show_progress_bar=True)
    np.savez(cache_path, embeddings=embeddings, sentence_hash=expected_hash)
    print(f"[cache] saved to {cache_path}")
    return embeddings
