"""
Dimensionality reduction (UMAP) + clustering (HDBSCAN).

Not plain k-means, not plain HDBSCAN on raw embeddings -- both were
tried and failed. See devlog 2026-08-17 for the full reasoning: k-means
forces every point into one of k clusters, silently absorbing real-but-
rare themes into a larger generic cluster; plain HDBSCAN on 768-dim
embeddings suffers the curse of dimensionality (giant blob or ~100%
noise depending on tuning). UMAP down to 5 dimensions first, then
HDBSCAN, is the combination that actually separated a real 29-sentence
theme from a much larger surrounding cluster.
"""
import numpy as np
import umap
from sklearn.cluster import HDBSCAN

from .config import DEFAULT_CONFIG, PipelineConfig


def cluster_embeddings(embeddings: np.ndarray, config: PipelineConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Returns an array of cluster labels, one per input embedding.
    -1 means "noise" -- didn't fit any cluster densely enough, not
    forced into a false grouping."""
    n_neighbors = min(config.umap_max_neighbors, len(embeddings) - 1)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=config.umap_n_components,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        random_state=config.random_state,
    )
    reduced = reducer.fit_transform(embeddings)

    hdb = HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        metric=config.hdbscan_metric,
    )
    return hdb.fit_predict(reduced)
