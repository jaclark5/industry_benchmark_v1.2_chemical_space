"""Supporting functions for plotting the PCA of clustered fingerprints / similarity methods"""

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from openeye.oegraphsim import OEFingerPrint


def get_final_contenders(
    filename: str,
    noise_cut: float = 0.5,
    n_clusters_cut: int = 10,
) -> list[tuple[str, str, int, int]]:
    """
    Pull strings identifying final contenders from a CSV file given cutoff numbers of clusters and noise fractions.

    Parameters
    ----------
    filename : str
        Path to the CSV file containing the results.
    noise_cut : float, optional
        Maximum allowed noise fraction. Default is 0.5.
    n_clusters_cut : int, optional
        Minimum number of clusters. Default is 10.

    Returns
    -------
    contenders : list[tuple[str]]
        List of tuples containing similarity method, fingerprint type, minimum cluster size, and minimum samples.
    """

    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    filtered_df = df[
        (df["Noise Fraction"] <= noise_cut) & (df["Num Clusters"] >= n_clusters_cut)
    ]
    contenders = [
        (
            row["Similarity Method"],
            row["Fingerprint Type"],
            row["Min Cluster Size"],
            row["Min Samples"],
        )
        for _, row in filtered_df.iterrows()
    ]

    return contenders


def get_cluster_labels(filename: str) -> list[int]:
    """
    Retrieve cluster labels from a file.

    Parameters
    ----------
    filename : str
        Path to the file containing cluster labels.

    Returns
    -------
    list[int]
        List of cluster labels.
    """
    with open(filename, "r") as f:
        labels = [int(line.strip()) for line in f if line.strip()]

    return labels


def fingerprints_to_numpy_dense(fps):
    """
    Convert a list of OpenEye OEFingerPrint objects to a dense numpy array.

    Parameters
    ----------
    fps : list of OEFingerPrint
        List of fingerprint objects (all same size).

    Returns
    -------
    numpy.ndarray
        Array of shape (n_mols, n_bits) with dtype=np.uint8.
    """

    n_mols = len(fps)
    n_bits = fps[0].GetSize()

    X = np.zeros((n_mols, n_bits), dtype=np.uint8)
    for i, fp in enumerate(fps):
        X[i, :] = np.array([fp.IsBitOn(ii) for ii in range(n_bits)], dtype=int)

    return X


def plot_pca(
    fingerprints_clustered: list[OEFingerPrint],
    fingerprints_removed: list[OEFingerPrint],
    filename_plot: str,
    cluster_label_dict: dict[str, list[int]],
) -> None:
    """
    Perform PCA on clustered fingerprints and plot the results.

    Parameters
    ----------
    fingerprints_clustered : list[OEFingerPrint]
        List of OpenEye fingerprint objects for clustered fingerprints.
    fingerprints_removed : list[OEFingerPrint]
        List of OpenEye fingerprint objects for removed fingerprints.
    filename_plot : str
        Path to save the PCA plot.
    cluster_label_dict : dict[int, list[int]]
        Dictionary mapping cluster labels to indices.

    Returns
    -------
    None
    """

    X_clustered = fingerprints_to_numpy_dense(fingerprints_clustered)
    X_removed = fingerprints_to_numpy_dense(fingerprints_removed)

    # Perform PCA on clustered fingerprints
    pca = PCA(n_components=2)
    X_clustered_pca = pca.fit_transform(X_clustered)
    X_removed_pca = pca.transform(X_removed)

    # Prepare colors for clusters (exclude black)
    for cluster_label, cluster_indices in cluster_label_dict.items():
        if len(cluster_indices) != np.shape(X_clustered_pca)[0]:
            raise ValueError("Length of cluster_indices must match X_clustered_pca")
        cluster_dict = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_indices):
            cluster_dict[cluster_id].append(X_clustered_pca[idx])

        num_clusters = len(cluster_dict)
        palette = sns.color_palette("hsv", num_clusters)
        plt.figure(figsize=(8, 6))

        pca_values = np.array(cluster_dict[-1])
        plt.scatter(
            pca_values[:, 0],
            pca_values[:, 1],
            s=20,
            color="gray",
            label="Noise",
            alpha=0.7,
            edgecolor="none",
        )

        for i, (cluster_id, pca_values) in enumerate(cluster_dict.items()):
            if cluster_id == -1:
                continue
            pca_values = np.array(pca_values)

            plt.scatter(
                pca_values[:, 0],
                pca_values[:, 1],
                s=20,
                color=palette[i],
                label=None,
                alpha=0.5,
                edgecolor="none",
            )

        plt.scatter(
            X_removed_pca[:, 0],
            X_removed_pca[:, 1],
            s=45,
            color="black",
            label="Removed Molecules",
            alpha=1,
            edgecolor="none",
        )

        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend(loc="best")
        plt.tight_layout()
        ext = f".{filename_plot.split('.')[1]}"
        plt.savefig(filename_plot.replace(ext, f"_{cluster_label}{ext}"), dpi=600)
        plt.close()
