"""Dimensionality reduction utilities for embedding precomputed molecular distance matrices.

Why UMAP, MDS, and t-SNE instead of PCA
-----------------------------------------
PCA finds the axes of maximum *Euclidean* variance in the raw feature space (fingerprint
bit-vectors). The clustering in this pipeline, however, was computed using Tanimoto, Dice,
or Cosine distances — metrics that are defined on the fingerprint *intersection / union*
structure and are non-Euclidean. These two spaces can disagree sharply: two molecules that
are near-neighbours by Tanimoto distance may appear far apart in the Euclidean PCA
projection, and vice versa. As a result, HDBSCAN clusters will not in general appear as
compact, visually separated groups in a PCA scatter plot.

The three methods implemented here all accept a *precomputed* distance matrix directly,
so the 2D embedding is built from exactly the same pairwise distances that HDBSCAN used:

UMAP (Uniform Manifold Approximation and Projection)
    Graph-based method that constructs a weighted kNN graph from the distance matrix and
    optimises a low-dimensional layout that preserves both local neighbourhood structure and
    global topology. Fast, scales well, and tends to produce the most visually interpretable
    cluster separation. Recommended first choice.
    Usage: ``umap.UMAP(metric="precomputed")``

MDS (Metric / Classical Multidimensional Scaling)
    Finds a 2D configuration that minimises the stress between pairwise distances in the
    embedding and the original distance matrix. Preserves large-scale distance relationships
    well but is O(n³) so it scales poorly past ~3,000 molecules.
    Usage: ``sklearn.manifold.MDS(dissimilarity="precomputed")``

t-SNE (t-Distributed Stochastic Neighbour Embedding)
    Probability-based method that preserves local neighbourhood structure. Particularly good
    at revealing fine-grained cluster sub-structure but does not preserve global distances.
    Usage: ``sklearn.manifold.TSNE(metric="precomputed")``

All three approaches guarantee that the visualisation geometry is consistent with the
clustering geometry. PCA on bit-vectors does not provide this guarantee.
"""

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.manifold import MDS, TSNE
import umap


SUPPORTED_METHODS = ["UMAP", "MDS", "TSNE"]


def embed_distances(
    dist_matrix: np.ndarray,
    methods: list[str] = SUPPORTED_METHODS,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """
    Compute 2D embeddings of a precomputed distance matrix using one or more methods.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Square symmetric precomputed distance matrix of shape (n_samples, n_samples).
    methods : list[str], optional
        Reduction methods to apply. Any subset of ``["UMAP", "MDS", "TSNE"]``.
        Defaults to all three.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from method name to (n_samples, 2) embedding array.

    Raises
    ------
    ImportError
        If ``"UMAP"`` is requested but ``umap-learn`` is not installed.
    ValueError
        If an unrecognised method name is given.
    """
    embeddings: dict[str, np.ndarray] = {}

    for method in methods:
        logger.info(
            f"  Computing {method} embedding ({dist_matrix.shape[0]} samples)..."
        )

        if method == "UMAP":
            reducer = umap.UMAP(
                n_components=2,
                metric="precomputed",
                random_state=random_state,
            )
            embeddings[method] = reducer.fit_transform(dist_matrix)

        elif method == "MDS":
            mds = MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=random_state,
                n_jobs=-1,
            )
            embeddings[method] = mds.fit_transform(dist_matrix)

        elif method == "TSNE":
            tsne = TSNE(
                n_components=2,
                metric="precomputed",
                random_state=random_state,
                init="random",
            )
            embeddings[method] = tsne.fit_transform(dist_matrix)

        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: {SUPPORTED_METHODS}"
            )

        logger.info(f"  {method} embedding done.")

    return embeddings


def plot_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    filename: str,
    title: str,
    noise_label: int = -1,
) -> None:
    """
    Save a 2D scatter plot of a dimensionality-reduction embedding coloured by cluster labels.

    Noise points (label == ``noise_label``) are drawn in light grey behind the clusters.
    Each unique cluster ID is assigned a distinct colour from a perceptually-uniform
    colourmap; with more than 20 clusters, ``"turbo"`` is used for maximum discrimination.

    Parameters
    ----------
    embedding : np.ndarray
        (n_samples, 2) array of 2D coordinates.
    labels : np.ndarray
        Integer cluster labels of length n_samples. HDBSCAN noise points should be -1.
    filename : str
        Output file path (PNG).
    title : str
        Figure title.
    noise_label : int, optional
        Label value used for unclustered / noise points. Default is -1.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    noise_mask = labels == noise_label
    cluster_ids = sorted(set(labels[~noise_mask]))
    n_clusters = len(cluster_ids)

    # Draw noise behind cluster points
    if noise_mask.any():
        ax.scatter(
            embedding[noise_mask, 0],
            embedding[noise_mask, 1],
            c="lightgrey",
            s=5,
            linewidths=0,
            label=f"Noise (n={noise_mask.sum()})",
            zorder=1,
        )

    cmap = plt.cm.get_cmap("tab20" if n_clusters <= 20 else "turbo", max(n_clusters, 1))
    for idx, cid in enumerate(cluster_ids):
        mask = labels == cid
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[cmap(idx / max(n_clusters - 1, 1))],
            s=8,
            linewidths=0,
            zorder=2,
        )

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_xticks([])
    ax.set_yticks([])

    n_noise = int(noise_mask.sum())
    noise_pct = 100 * n_noise / len(labels)
    ax.text(
        0.01,
        0.01,
        f"Clusters: {n_clusters}  |  Noise: {n_noise} ({noise_pct:.1f}%)",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        color="dimgrey",
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
