"""Get cluster labels for each similarity / fingerprint combination and HDBSCAN hyperparameters

The major inputs of this module are:

- ../2_distances/outputs/distances_{Tanimoto | Dice | Cosine}_{Circular | MACCS | LINGO}.txt files
    Containing the full distance matrix (1 - similarity) for "OpenFF Industry Benchmark Season 1 v1.2"

The major outputs of this module are:

- outputs/distances_{Tanimoto | Dice | Cosine}_{Circular | MACCS | LINGO}_{min_cluster_size}_{min_samples}_cluster_size_hist.png
    Cluster sizes resulting from HDBSCAN hyperparameters
- outputs/distances_{Tanimoto | Dice | Cosine}_{Circular | MACCS | LINGO}_{min_cluster_size}_{min_samples}.txt
    Cluster labels resulting from HDBSCAN hyperparameters
- outputs/distances_{Tanimoto | Dice | Cosine}_{Circular | MACCS | LINGO}.png
    Heatmap of Number of Clusters and Noise Fraction plotted on axes min_samples and min_cluster_size
- outputs/summary.py
    CSV containing a summary of clustering performance given a fingerprint type, similarity method, min_cluster_size, or min_samples,
    reporting the number of clusters and noise fraction.

Outputs for a sparse matrix of each distance matrix with a KNN k=500
- outputs_sparse/distances_{Tanimoto | Dice | Cosine}_{Circular | MACCS | LINGO}_kvalues_elbow.png
    Plot of inertia and distortion for k-values of distance matrix
- outputs_sparse/distances_{Tanimoto | Dice | Cosine}_{Circular | MACCS | LINGO}_{min_cluster_size}_{min_samples}_cluster_size_hist.png
    Cluster sizes resulting from HDBSCAN hyperparameters
- outputs_sparse/distances_{Tanimoto | Dice | Cosine}_{Circular | MACCS | LINGO}_{min_cluster_size}_{min_samples}.txt
    Cluster labels resulting from HDBSCAN hyperparameters
- outputs_sparse/distances_{Tanimoto | Dice | Cosine}_{Circular | MACCS | LINGO}.png
    Heatmap of Number of Clusters and Noise Fraction plotted on axes min_samples and min_cluster_size
- outputs_sparse/summary.py
    CSV containing a summary of clustering performance given a fingerprint type, similarity method, min_cluster_size, or min_samples,
    reporting the number of clusters and noise fraction.


"""

import os
import glob

from loguru import logger
import numpy as np

from cluster_utilities import (
    tune_hdbscan,
    plot_k_distance,
    save_results,
    plot_results,
    plot_final_contenders,
)


logger.info(
    "_______________________________________________________________________________________________________________"
)
logger.info(
    "__________________________ Assess Clustering Parameters with FP and Similarity Types __________________________"
)
logger.info(
    "___________________________________________ with Sparse Matrices ______________________________________________"
)

os.makedirs("outputs_sparse", exist_ok=True)


files = glob.glob("../2_distances/outputs/distances_*.txt")
if os.path.isfile("outputs_sparse/summary.csv"):
    os.remove("outputs_sparse/summary.csv")

for i, file in enumerate(files):
    filename = f"outputs_sparse/{os.path.split(file)[1].split('.')[0]}.png"
    _, similarity_type, fingerprint_type = (
        os.path.split(file)[1].split(".")[0].split("_")
    )

    logger.info(f"Running {file}, {i+1} of {len(files)}")
    logger.info("Importing data...")
    dist_matrix = np.genfromtxt(file, delimiter=",").astype(np.float32)
    logger.info("Check k values...")
    if not os.path.isfile(filename.replace(".png", "_kvalues.png")):
        plot_k_distance(dist_matrix, filename.replace(".png", "_kvalues.png"))
    logger.info("Running hyperparameter variations...")
    results = tune_hdbscan(dist_matrix, filename.split(".")[0] + ".txt", sparse=True)
    logger.info("Saving results...")
    save_results(
        results, similarity_type, fingerprint_type, "outputs_sparse/summary.csv"
    )
    logger.info("Plotting results...")
    plot_results(results, filename)

plot_final_contenders(
    "outputs_sparse/summary.csv",
    filename_plot="outputs_sparse/filtered_comparison_plot.png",
)


logger.info(
    "_______________________________________________________________________________________________________________"
)
logger.info(
    "__________________________ Assess Clustering Parameters with FP and Similarity Types __________________________"
)
logger.info(
    "___________________________________________ with Full Matrices ______________________________________________"
)

os.makedirs("outputs", exist_ok=True)


files = glob.glob("../2_distances/outputs/distances_*.txt")
if os.path.isfile("outputs/summary.csv"):
    os.remove("outputs/summary.csv")
for i, file in enumerate(files):
    filename = f"outputs/{os.path.split(file)[1].split('.')[0]}.png"
    _, similarity_type, fingerprint_type = (
        os.path.split(file)[1].split(".")[0].split("_")
    )

    logger.info(f"Running {file}, {i+1} of {len(files)}")
    logger.info("Importing data...")
    dist_matrix = np.genfromtxt(file, delimiter=",").astype(np.float32)
    logger.info("Running hyperparameter variations...")
    results = tune_hdbscan(dist_matrix, filename.split(".")[0] + ".txt")
    logger.info("Saving results...")
    save_results(results, similarity_type, fingerprint_type, "outputs/summary.csv")
    logger.info("Plotting results...")
    plot_results(results, filename)

plot_final_contenders(
    "outputs/summary.csv", filename_plot="outputs/filtered_comparison_plot.png"
)
