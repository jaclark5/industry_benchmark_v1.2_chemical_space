"""Supporting functions for clustering fingerprints using similarity analysis"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from loguru import logger
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors


def plot_k_distance(
    X: np.ndarray, filename: str, ks: list[int] = list(range(2, 51))
) -> None:
    """
    Plot distortion and inertia metrics for a range of k values using the elbow method, and save the plot to a file.

    Parameters
    ----------
    X : np.ndarray
        Precomputed distance matrix of shape (n_samples, n_samples).
    filename : str
        Path to save the generated plot (PNG format).
    ks : list[int], optional
        List of k values (number of clusters) to evaluate. Defaults to range(2, 51).

    Returns
    -------
    None
    """
    distortions = []
    inertias = []
    for k in ks:
        kmeanModel = KMeans(n_clusters=k, random_state=42).fit(X)

        distortions.append(
            sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1) ** 2)
            / X.shape[0]
        )
        inertias.append(kmeanModel.inertia_)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "tab:blue"
    ax1.set_xlabel("k")
    ax1.set_ylabel("Distortion", color=color1)
    ax1.plot(ks, distortions, marker="o", color=color1, label="Distortion")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("Inertia", color=color2)
    ax2.plot(ks, inertias, marker="s", color=color2, linestyle="--", label="Inertia")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Elbow Method: Distortion & Inertia vs k")

    # Elbow detection (simple heuristic: max second derivative for inertia)
    inertia_diff2 = np.diff(inertias, 2)
    if len(inertia_diff2) > 0:
        elbow_idx = np.argmin(inertia_diff2) + 1  # one index after the max
        ax2.axvline(
            ks[elbow_idx],
            color="r",
            linestyle="--",
            label=f"Elbow at k={ks[elbow_idx]}",
        )

    lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
    lines: list = []
    labels: list = []
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax1.legend(lines, labels, loc="upper left")
    fig.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()


def build_mutual_knn_sparse(dist_mat: np.ndarray, k: int = 500) -> csr_matrix:
    """
    Create a sparse, symmetric k-nearest neighbors distance matrix.

    Parameters
    ----------
    dist_mat : np.ndarray
        Full (n, n) numpy array of distances (1 - similarity).
    k : int, optional
        Number of nearest neighbors to consider. Default is 500.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse mutual k-nearest neighbors matrix (float32).
    """
    nbrs = NearestNeighbors(n_neighbors=k, metric="precomputed").fit(dist_mat)
    knn_graph = nbrs.kneighbors_graph(dist_mat, mode="distance")
    # Make mutual: keep edges where both directions exist
    # Make symmetric: ensure each point has at least k neighbors (if edge exists in either direction, include it)
    mutual = knn_graph.maximum(knn_graph.transpose())
    # Ensure diagonal is zero
    mutual.setdiag(0)
    mutual.eliminate_zeros()
    # Convert to float32
    mutual = mutual.astype(np.float32)
    return mutual


def run_hdbscan(
    dist_matrix: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.1,
    sparse: bool = False,
) -> np.ndarray:
    """
    Cluster using HDBSCAN with a precomputed distance matrix.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Precomputed pairwise distance matrix.
    min_cluster_size : int, optional
        Minimum size of clusters. Start around the expected smallest chemical family size you want to detect. Default is 10.
    min_samples : int, optional
        Minimum samples in a cluster. Start at min_samples ≈ μ/σ * 2 (rough heuristic: bigger spread → lower value).
        Default is 5.
    cluster_selection_epsilon : float, optional
        Distance threshold for cluster selection. If many clusters are breaking apart at distances slightly above μ, set
        this to smooth it over. Default is 0.1
    sparse : bool, optional
        If True, the distance matrix is made sparse using :func:`build_mutual_knn_sparse`

    Returns
    -------
    np.ndarray
        Array of cluster labels for each point.

    Notes
    -----
    - ``cluster_selection_method="leaf"``
    """
    clusterer = HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method="leaf",
    )
    if sparse:
        sparse_matrix = build_mutual_knn_sparse(dist_matrix.astype(np.float32))
        labels = clusterer.fit_predict(sparse_matrix)
    else:
        labels = clusterer.fit_predict(dist_matrix.astype(np.float32))

    return labels


def tune_hdbscan(
    dist_matrix: np.ndarray,
    filename: str,
    min_cluster_sizes: list[int] = [10, 20, 50, 100],
    min_samples_list: list[int] = [1, 2, 3, 5, 10, 15],
    sparse: bool = False,
) -> list[dict[str, float]]:
    """
    Perform grid search over HDBSCAN hyperparameters and record clustering results.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Precomputed pairwise distance matrix.
    filename : str
        Base filename for output label files.
    min_cluster_sizes : list[int], optional
        Candidate min_cluster_size values to try. Default is [10, 20, 50, 100].
    min_samples_list : list[int], optional
        Candidate min_samples values to try. Default is [1, 2, 3, 5, 10, 15].
    sparse : bool, optional
        If True, use a sparse k-nearest neighbors matrix.

    Returns
    -------
    list[dict[str, float]]
        Each entry contains keys: "min_cluster_size", "min_samples", "n_clusters", "noise_frac".
    """
    n = dist_matrix.shape[0]
    results = []

    for mcs in min_cluster_sizes:
        for ms in min_samples_list:
            logger.info(f"    Running {mcs} {ms}")
            labels = run_hdbscan(
                dist_matrix,
                min_cluster_size=mcs,
                min_samples=ms,
                sparse=sparse,
            )
            tmp_filename = f"{filename.split('.')[0]}_{mcs}_{ms}.txt"
            open(tmp_filename, "w").write("\n".join([str(x) for x in labels]))
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_frac = np.sum(labels == -1) / n
            results.append(
                {
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                    "n_clusters": n_clusters,
                    "noise_frac": noise_frac,
                }
            )
            logger.info(
                f"mcs={mcs}, ms={ms} -> clusters={n_clusters}, noise={noise_frac:.2f}"
            )

            # Save histogram of cluster sizes
            cluster_sizes = [np.sum(labels == i) for i in set(labels) if i != -1]
            plt.figure(figsize=(8, 5))
            plt.hist(cluster_sizes, bins=30, color="skyblue", edgecolor="black")
            plt.xlabel("Cluster Size")
            plt.ylabel("Count")
            plt.title(f"Distribution of Cluster Sizes (mcs={mcs}, ms={ms})")
            plt.tight_layout()
            plt.savefig(tmp_filename.replace(".txt", "_cluster_size_hist.png"), dpi=600)
            plt.close()

    return results


def save_results(
    results: list[dict[str, float]],
    similarity_method: str,
    fingerprint_type: str,
    filename: str,
    mode: str = "a",
) -> None:
    """
    Save results of HDBSCAN tuning to a CSV file.

    Parameters
    ----------
    results : list[dict[str, float]]
        Results from HDBSCAN tuning.
    similarity_method : str
        Name of the similarity method used.
    fingerprint_type : str
        Type of fingerprint used.
    filename : str
        Filename for output summary.
    mode : str, optional
        File write mode ('a' for append, 'w' for overwrite). Default is 'a'.

    Returns
    -------
    None
    """
    header = True if not os.path.isfile(filename) or mode == "w" else False
    with open(filename, mode) as f:
        if header:
            f.write(
                "Fingerprint Type, Similarity Method, Min Cluster Size, Min Samples, Num Clusters, Noise Fraction\n"
            )
        for result in results:
            f.write(
                ", ".join(
                    [fingerprint_type, similarity_method]
                    + [str(x) for x in result.values()]
                )
                + "\n"
            )


def plot_results(results: list[dict[str, float]], filename: str) -> None:
    """
    Plot results of HDBSCAN tuning as heatmaps for number of clusters and noise fraction.

    Parameters
    ----------
    results : list[dict[str, float]]
        Results from HDBSCAN tuning.
    filename : str
        Filename for output plots.

    Returns
    -------
    None
    """
    # Convert to grid for heatmaps
    mcs_vals = sorted(set(r["min_cluster_size"] for r in results))
    ms_vals = sorted(set(r["min_samples"] for r in results))
    cluster_grid = np.zeros((len(mcs_vals), len(ms_vals)))
    noise_grid = np.zeros((len(mcs_vals), len(ms_vals)))

    for r in results:
        i = mcs_vals.index(r["min_cluster_size"])
        j = ms_vals.index(r["min_samples"])
        cluster_grid[i, j] = r["n_clusters"]
        noise_grid[i, j] = r["noise_frac"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im1 = axes[0].imshow(cluster_grid, cmap="viridis", aspect="auto")
    axes[0].set_xticks(range(len(ms_vals)))
    axes[0].set_xticklabels(ms_vals)
    axes[0].set_yticks(range(len(mcs_vals)))
    axes[0].set_yticklabels(mcs_vals)
    axes[0].set_xlabel("min_samples")
    axes[0].set_ylabel("min_cluster_size")
    axes[0].set_title("Number of clusters")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(noise_grid, cmap="magma_r", aspect="auto")
    axes[1].set_xticks(range(len(ms_vals)))
    axes[1].set_xticklabels(ms_vals)
    axes[1].set_yticks(range(len(mcs_vals)))
    axes[1].set_yticklabels(mcs_vals)
    axes[1].set_xlabel("min_samples")
    axes[1].set_ylabel("min_cluster_size")
    axes[1].set_title("Noise fraction")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()


def plot_final_contenders(
    filename: str,
    noise_cut: float = 0.5,
    n_clusters_cut: int = 10,
    filename_plot: str = "filtered_comparison_plot.png",
) -> None:
    """
    Plot final contenders from a CSV file as a scatterplot of clusters vs noise fraction.

    Parameters
    ----------
    filename : str
        Path to the CSV file containing the results.
    noise_cut : float, optional
        Maximum allowed noise fraction. Default is 0.5.
    n_clusters_cut : int, optional
        Minimum number of clusters. Default is 10.
    filename_plot : str, optional
        Filename for the output plot. Default is "filtered_comparison_plot.png".

    Returns
    -------
    None
    """
    # Load data
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()

    # Filter data
    filtered_df = df[
        (df["Noise Fraction"] <= noise_cut) & (df["Num Clusters"] >= n_clusters_cut)
    ]

    # Create scatterplot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=filtered_df,
        x="Num Clusters",
        y="Noise Fraction",
        hue="Similarity Method",
        style="Fingerprint Type",
        size="Min Cluster Size",
        sizes=(50, 200),
        palette="colorblind",
        edgecolor="black",
        linewidth=0.5,
    )

    # Customize plot
    plt.title("Final Contenders: Clusters vs Noise Fraction")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Noise Fraction")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()

    # Save plot
    plt.savefig(filename_plot, dpi=600)
    plt.close()

    # Find rows not in filtered_df
    excluded_df = df.loc[~df.index.isin(filtered_df.index)]

    for _, row in excluded_df.iterrows():
        sim_method = str(row["Similarity Method"]).strip()
        fp_type = str(row["Fingerprint Type"]).strip()
        min_cluster_size = str(row["Min Cluster Size"]).strip()
        min_samples = str(row["Min Samples"]).strip()
        plot_name = f"distances_{sim_method}_{fp_type}_{min_cluster_size}_{min_samples}_cluster_size_hist.png"
        plot_path = os.path.join(os.path.dirname(filename_plot), plot_name)
        if os.path.isfile(plot_path):
            os.remove(plot_path)
