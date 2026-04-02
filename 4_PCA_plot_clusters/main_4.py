"""Visualise HDBSCAN clustering results using distance-matrix-aware dimensionality reduction.

The major inputs of this module are:

- ../2_distances/outputs/distances_{Tanimoto | Dice | Cosine}_{Circular | MACCS | Lingo}.txt
    Full pairwise distance matrices (1 - similarity) for the Industry Benchmark v1.2 molecules.
- ../3_cluster/outputs/summary.csv
    HDBSCAN hyperparameter sweep results for full distance matrices.
- ../3_cluster/outputs_sparse/summary.csv
    HDBSCAN hyperparameter sweep results for sparse kNN distance matrices.
- ../3_cluster/outputs/distances_{sim}_{fp}_{mcs}_{ms}.txt
    Cluster labels for each full-matrix contender.
- ../3_cluster/outputs_sparse/distances_{sim}_{fp}_{mcs}_{ms}.txt
    Cluster labels for each sparse-matrix contender.

The major outputs of this module are (under outputs/ and outputs_sparse/):

- outputs/{full|sparse}_{UMAP|MDS|TSNE}_{sim}_{fp}_{mcs}_{ms}.png
    2D scatter plot coloured by HDBSCAN cluster label. One plot per method per contender.

Why UMAP, MDS, and t-SNE instead of PCA
-----------------------------------------
PCA finds axes of maximum *Euclidean* variance in the raw fingerprint bit-vector space.
The HDBSCAN clustering, however, was computed from Tanimoto, Dice, or Cosine distances —
non-Euclidean metrics that measure structural overlap, not Euclidean proximity.

A pair of molecules that are close neighbours by Tanimoto distance need not be close in
Euclidean PCA space, so HDBSCAN clusters will appear fragmented and overlapping in a PCA
plot. The three methods used here all accept the *precomputed* distance matrix directly:

UMAP  – builds a kNN graph from the distance matrix and optimises a layout that preserves
        both local neighbourhood and global topology; fast and visually interpretable.
MDS   – minimises the stress between pairwise distances in 2D and those in the original
        matrix; preserves large-scale relationships but is O(n³).
t-SNE – probability-based method that preserves local structure; good for revealing
        sub-cluster organisation but does not preserve global distances.
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger

from plot_utilities import SUPPORTED_METHODS, embed_distances, plot_embedding

# ── Configuration ────────────────────────────────────────────────────────────
# Methods to run. Any subset of SUPPORTED_METHODS = ["UMAP", "MDS", "TSNE"].
METHODS: list[str] = SUPPORTED_METHODS

# Contender filter thresholds (must match step 3 conventions)
NOISE_CUT: float = 0.5
N_CLUSTERS_CUT: int = 10
# ─────────────────────────────────────────────────────────────────────────────


def _load_labels(path: str) -> np.ndarray:
    """Read a one-integer-per-line cluster label file into a NumPy array."""
    return np.array([int(x) for x in open(path).read().strip().splitlines()])


def _get_contenders(summary_csv: str) -> pd.DataFrame:
    """Load summary CSV and return rows that pass the noise / cluster-count cuts."""
    df = pd.read_csv(summary_csv)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    df["Min Cluster Size"] = df["Min Cluster Size"].astype(int)
    df["Min Samples"] = df["Min Samples"].astype(int)
    return df[
        (df["Noise Fraction"] <= NOISE_CUT) & (df["Num Clusters"] >= N_CLUSTERS_CUT)
    ].reset_index(drop=True)


def run(
    summary_csv: str,
    label_dir: str,
    dist_matrix_dir: str,
    output_dir: str,
    methods: list[str] = METHODS,
) -> None:
    """
    Compute embeddings for all contenders in a summary CSV and save scatter plots.

    Embeddings are computed once per unique (similarity method, fingerprint type) pair and
    then reused for every set of cluster labels that shares that distance matrix.

    Parameters
    ----------
    summary_csv : str
        Path to the step-3 summary CSV.
    label_dir : str
        Directory containing cluster label ``.txt`` files from step 3.
    dist_matrix_dir : str
        Directory containing distance matrix ``.txt`` files from step 2.
    output_dir : str
        Directory to write embedding scatter plots into.
    methods : list[str], optional
        Dimensionality reduction methods to apply. Default is all three.
    """
    os.makedirs(output_dir, exist_ok=True)

    contenders = _get_contenders(summary_csv)
    if contenders.empty:
        logger.warning(f"No contenders found in {summary_csv} — skipping.")
        return

    logger.info(f"Found {len(contenders)} contenders in {summary_csv}")

    # Group contenders by (fp_type, sim_type) so we load/embed each distance matrix once
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for _, row in contenders.iterrows():
        key = (row["Fingerprint Type"], row["Similarity Method"])
        groups[key].append(row)

    for (fp_type, sim_type), rows in groups.items():
        dist_path = os.path.join(dist_matrix_dir, f"distances_{sim_type}_{fp_type}.txt")
        if not os.path.isfile(dist_path):
            logger.warning(f"Distance matrix not found: {dist_path} — skipping group.")
            continue

        logger.info(f"Loading distance matrix: {dist_path}")
        dist_matrix = np.genfromtxt(dist_path, delimiter=",").astype(np.float32)

        logger.info(f"Embedding {fp_type}/{sim_type} with methods: {methods}")
        embeddings = embed_distances(dist_matrix, methods=methods)

        for row in rows:
            mcs = row["Min Cluster Size"]
            ms = row["Min Samples"]
            label_path = os.path.join(
                label_dir, f"distances_{sim_type}_{fp_type}_{mcs}_{ms}.txt"
            )
            if not os.path.isfile(label_path):
                logger.warning(f"Label file not found: {label_path} — skipping.")
                continue

            labels = _load_labels(label_path)
            n_clusters = int(row["Num Clusters"])
            noise_pct = float(row["Noise Fraction"]) * 100

            for method, embedding in embeddings.items():
                out_path = os.path.join(
                    output_dir,
                    f"{method}_{sim_type}_{fp_type}_{mcs}_{ms}.png",
                )
                title = (
                    f"{method}  |  {fp_type} / {sim_type}"
                    f"  |  mcs={mcs} ms={ms}"
                    f"  |  {n_clusters} clusters  {noise_pct:.0f}% noise"
                )
                logger.info(f"  Plotting {out_path}")
                plot_embedding(embedding, labels, out_path, title)


# ── Full matrices ─────────────────────────────────────────────────────────────
logger.info(
    "_______________________________________________________________________________________________________________"
)
logger.info(
    "_________________________________ Plot Embeddings with Full Distance Matrices _________________________________"
)

run(
    summary_csv="../3_cluster/outputs/summary.csv",
    label_dir="../3_cluster/outputs",
    dist_matrix_dir="../2_distances/outputs",
    output_dir="outputs/full",
    methods=METHODS,
)

# ── Sparse matrices ───────────────────────────────────────────────────────────
logger.info(
    "_______________________________________________________________________________________________________________"
)
logger.info(
    "_________________________________ Plot Embeddings with Sparse Distance Matrices ________________________________"
)

run(
    summary_csv="../3_cluster/outputs_sparse/summary.csv",
    label_dir="../3_cluster/outputs_sparse",
    dist_matrix_dir="../2_distances/outputs",
    output_dir="outputs/sparse",
    methods=METHODS,
)
