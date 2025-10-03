"""Distance matrices using various similarity methods with available fingerprints.

The major outputs of this module are in the outputs directory:

- outputs/distance_distributions.png
    Plotted distributions of similarity values based on method and fingerprint input.
- outputs/distances_{Tanimoto | Dice | Cosine}_{Circular | MACCS | LINGO}.txt files
    The first line contains the number of molecules and subsequent rows containing the upper triangle arrays for
    "OpenFF Industry Benchmark Season 1 v1.2"
- outputs/removed_distance_{Tanimoto | Dice | Cosine}_{Circular | MACCS | LINGO}.txt files
    Each row contains the distance between that molecule and all molecules in the Industry Benchmark v1.2.

"""

import os
import sys
from collections import defaultdict

from loguru import logger

from common_molecule_utilities import (
    get_molecules,
    get_cluster_ref_mols,
    print_structures,
)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../4_PCA_plot_clusters"))
)
from pca_cluster_utilities import get_cluster_labels  # noqa: E402


os.makedirs("outputs", exist_ok=True)

logger.info("Get clusters for selected final contenders")
selected_contenders = {
    "sparse": [("DICE", "MACCS", 20, 2)],
    "full": [
        ("Tanimoto", "MACCS", 10, 10),
        ("DICE", "Circular", 10, 2),
        ("Tanimoto", "Circular", 10, 2),
    ],
}
cluster_label_types = {}
for matrix_type, contenders in selected_contenders.items():
    path = (
        "../3_cluster/outputs/"
        if matrix_type == "full"
        else "../3_cluster/outputs_sparse/"
    )
    label = "full" if matrix_type == "full" else "sparse"
    for sim_method, fp_type, min_clust_size, min_samples in contenders:
        labels = get_cluster_labels(
            path
            + f"distances_{sim_method}_{fp_type}_{min_clust_size}_{min_samples}.txt"
        )
        cluster_label_types[
            f"{label}_{sim_method}_{fp_type}_{min_clust_size}_{min_samples}"
        ] = labels

logger.info("Get molecules")
mols = get_molecules(
    "../1_import_data_process_fps/outputs/industry_benchmark_v1.2_unique_mols.sdf"
)
n_atoms = list(set(x.NumAtoms() for x in mols))
if len(n_atoms) == 1 and n_atoms[0] == 0:
    raise ValueError("Molecules did not import properly!")

logger.info("Get greatest common structure for clusters and print clusters")
for label, cluster_labels in cluster_label_types.items():
    if len(mols) != len(cluster_labels):
        raise ValueError(
            "Number of cluster labels should equal the number of molecules"
        )

    cluster_dict = defaultdict(list)
    for cluster_id, mol in zip(cluster_labels, mols):
        cluster_dict[cluster_id].append(mol)

    for cluster_id, tmp_mols in cluster_dict.items():
        if len(tmp_mols) > 300:
            continue
        logger.info(f"Printing PDF of molecules in {label}, cluster {cluster_id}")
        print_structures(tmp_mols, f"outputs/{label}_{cluster_id}.pdf")

    logger.info(f"Getting ref molecules {len(cluster_dict)} for {label}")
    del cluster_dict[-1]
    cluster_dict = defaultdict(
        list, sorted(cluster_dict.items())
    )  # Ensure type consistency
    ref_mols = get_cluster_ref_mols(cluster_dict, f"outputs/{label}.csv")

    logger.info("Saving pdf of ref molecules...")
    print_structures(ref_mols, f"outputs/{label}.pdf")
