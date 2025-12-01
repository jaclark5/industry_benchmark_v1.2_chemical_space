"""Distance matrices using various similarity methods with available fingerprints.

The major outputs of this module are in the outputs directory:

- outputs/distances_distributions.png
    Plotted distributions of similarity values based on method and fingerprint input.
- outputs/distances_{Tanimoto | Dice | Cosine}_{Circular | MACCS | LINGO}.txt files
    The first line contains the number of molecules and subsequent rows containing the upper triangle arrays for
    "OpenFF Industry Benchmark Season 1 v1.2"
- outputs/removed_distances_{Tanimoto | Dice | Cosine}_{Circular | MACCS | LINGO}.txt files
    Each row contains the distance between that molecule and all molecules in the Industry Benchmark v1.2.

"""

import os
import sys
from collections import defaultdict

from loguru import logger

from pca_cluster_utilities import plot_pca, get_cluster_labels, get_final_contenders

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../2_distances"))
)
from distance_utilities import get_fingerprints  # noqa: E402

os.makedirs("outputs", exist_ok=True)

logger.info("Getting Industry Benchmark fingerprints")
fps_ib = get_fingerprints(
    "../1_import_data_process_fps/outputs/industry_benchmark_v1.2_unique_mols.sdf"
)
logger.info("Getting fingerprints of removed molecules")
fps_removed = get_fingerprints(
    "../1_import_data_process_fps/outputs/removed_records_unique_mols.sdf"
)

logger.info(
    "_______________________________________________________________________________________________________________"
)
logger.info(
    "_________________________________ Plot PCA with Clustering with Full Matrices _________________________________"
)

contenders = get_final_contenders("../3_cluster/outputs/summary.csv")
logger.info(f"Obtained {len(contenders)} final contenders from full matrix clustering")
cluster_label_types: defaultdict[str, dict[str, list[int]]] = defaultdict(dict)
for sim_method, fp_type, min_clust_size, min_samples in contenders:
    labels = get_cluster_labels(
        f"../3_cluster/outputs/distances_{sim_method}_{fp_type}_{min_clust_size}_{min_samples}.txt"
    )
    cluster_label_types[fp_type][
        f"{sim_method}_{min_clust_size}_{min_samples}"
    ] = labels

logger.info("Plotting PCA for each clustering output...")
for fp_type, cluster_labels in cluster_label_types.items():
    logger.info(
        f"Clustering {len(cluster_labels)} variations for Fingerprint {fp_type}"
    )
    plot_pca(
        fps_ib[fp_type],
        fps_removed[fp_type],
        f"outputs/pca_{fp_type}.png",
        cluster_labels,
    )


logger.info(
    "_______________________________________________________________________________________________________________"
)
logger.info(
    "_________________________________ Plot PCA with Clustering with Sparse Matrices _________________________________"
)

contenders = get_final_contenders("../3_cluster/outputs_sparse/summary.csv")
logger.info(
    f"Obtained {len(contenders)} final contenders from sparse matrix clustering"
)
for sim_method, fp_type, min_clust_size, min_samples in contenders:
    labels = get_cluster_labels(
        f"../3_cluster/outputs_sparse/distances_{sim_method}_{fp_type}_{min_clust_size}_{min_samples}.txt"
    )
    cluster_label_types[fp_type][
        f"{sim_method}_{min_clust_size}_{min_samples}"
    ] = labels

logger.info("Plotting PCA for each clustering output...")
for fp_type, cluster_labels in cluster_label_types.items():
    logger.info(
        f"Clustering {len(cluster_labels)} variations for Fingerprint {fp_type}"
    )
    plot_pca(
        fps_ib[fp_type],
        fps_removed[fp_type],
        f"outputs/sparse_pca_{fp_type}.png",
        cluster_labels,
    )
