"""Create dataframes for the Industry Benchmark v1.2 and Discarded molecules from v1.1, output various distance
matrixes.

The major outputs of this module are in the outputs directory:

- *.csv files
    Containing unique CMILES and QCArchive entry names from the "OpenFF Industry Benchmark Season 1 v1.1"
    for "OpenFF Industry Benchmark Season 1 v1.2" and removed entries.
- distances_{Tanimoto | Dice | Cosine}_{Circular | MACCS | LINGO}.txt files
    The first line contains the number of molecules and subsequent rows containing the upper triangle arrays for
    "OpenFF Industry Benchmark Season 1 v1.2"
- removed_distance_{Tanimoto | Dice | Cosine}_{Circular | MACCS | LINGO}.txt files
    Each row contains the distance between that molecule and all molecules in the Industry Benchmark v1.2.

"""

from urllib.request import urlopen
from loguru import logger
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from qcportal import PortalClient

from fp_utilities import get_fps, get_distance_matrix, get_distance_vector

os.makedirs("outputs", exist_ok=True)

# Pull Full Industry Dataset v1.1
ADDRESS = "https://api.qcarchive.molssi.org:443/"
qc_client = PortalClient(ADDRESS, cache_dir=".")
ds = qc_client.get_dataset("optimization", "OpenFF Industry Benchmark Season 1 v1.1")
logger.info(
    "Pulled optimization dataset 'OpenFF Industry Benchmark Season 1 v1.1' from QCArvive"
)


# Pull Records IDs that are habitually removed
with urlopen(
    "https://raw.githubusercontent.com/openforcefield/sage-2.2.0/refs/heads/main/05_benchmark_forcefield/process_bm/problem_ids/all_r7_outliers.txt"
) as response:
    file_content = response.read().decode()
outlier_record_ids = set([int(x) for x in file_content.splitlines()])
logger.info(
    f"There are {len(outlier_record_ids)} records that must be removed from Industry Benchmark v1.1 to make v1.2."
)

# Create dataframes for the cleaned Industry Dataset v1.2 and one for the removed entries:
filename_indbench = "outputs/industry_benchmark_v1.2_unique.csv"
filename_removed = "outputs/removed_records_unique.csv"
if not os.path.isfile(filename_indbench) or not os.path.isfile(filename_removed):
    removed_entries = []
    final_entries = []
    for entry_name, spec_name, rec in ds.iterate_records():
        if spec_name != "default":
            continue
        if rec.id in outlier_record_ids:
            removed_entries.append(
                {
                    "Name": entry_name,
                    "CMILES": rec.initial_molecule.extras[
                        "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                    ],
                }
            )
        else:
            final_entries.append(
                {
                    "Name": entry_name,
                    "CMILES": rec.initial_molecule.extras[
                        "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                    ],
                }
            )
    removed_df = pd.DataFrame(removed_entries)
    final_df = pd.DataFrame(final_entries)

    logger.info(
        f"For conformers, there are {len(final_df)} in v1.2 and {len(removed_df)} removed."
    )

    # Consolidate rows with the same CMILES
    removed_df = removed_df.groupby("CMILES", as_index=False).agg(
        {"Name": lambda x: list(x)}
    )
    removed_df.rename(columns={"Name": "Names"}, inplace=True)

    final_df = final_df.groupby("CMILES", as_index=False).agg(
        {"Name": lambda x: list(x)}
    )
    final_df.rename(columns={"Name": "Names"}, inplace=True)
    logger.info(
        f"For unique molecules, there are {len(final_df)} in v1.2 and {len(removed_df)} removed."
    )

    removed_df.to_csv(filename_removed, index=False)
    logger.info(f"Saved consolidated fingerprints '{filename_removed}'")
    final_df.to_csv(filename_indbench, index=False)
    logger.info(f"Saved consolidated fingerprints '{filename_indbench}'")
else:
    removed_df = pd.read_csv(filename_removed)
    final_df = pd.read_csv(filename_indbench)
    logger.info("Imported csv files")


# Calculate distance matrices of final_molecules
logger.info("Calculating distance matrices...")
fig, axs = plt.subplots(1, 3, figsize=(8, 5), sharey=True)
for j, fp_type in enumerate(["Circular", "MACCS", "LINGO"]):
    logger.info(f"Getting {fp_type} fingerprints")
    fps = get_fps(final_df["CMILES"].tolist(), fptype=fp_type)
    for i, sim_type in enumerate(["Tanimoto", "Dice", "Cosine"]):
        logger.info(f"Getting {sim_type} similarity distances")
        filename = f"outputs/distances_{sim_type}_{fp_type}.txt"
        if not os.path.isfile(filename):
            distances = get_distance_matrix(fps, sim_type)  # dist_list, npts
            logger.info(f"Saving 'outputs/distances_{sim_type}_{fp_type}.txt'")
            with open(filename, "w") as f:
                f.write(f"{distances[1]}\n")
                for n in distances[0]:
                    f.write(f"{n}\n")
        else:
            logger.info(f"Imported 'outputs/distances_{sim_type}_{fp_type}.txt'")
            data = [
                float(x.strip())
                for x in open(filename).read().strip().split("\n")
                if x != ""
            ]
            distances = (data[1:], int(data[0]))
        # Plot similarity distribution
        label = rf"{fp_type}: $\mu$ = {np.mean(distances[0]):0.2f} $\sigma$ = {np.std(distances[0]):0.2f}"
        axs[i].hist(
            distances[0],
            density=True,
            label=label,
            alpha=0.6,
            bins=30,
            edgecolor="black",
            linewidth=0.5,
        )
        if j == 0:
            if i == 0:
                axs[0].set_ylabel("Probability")
            axs[i].set_xlabel(f"{sim_type} Distance")
        axs[i].legend(loc="best")
        axs[i].set_xlim((0, 1))

plt.tight_layout()
fig.savefig("outputs/distance_distributions.png", dpi=600)
plt.close(fig)


# Save distances of removed molecules from final molecules
for fp_type in ["Circular", "MACCS", "LINGO"]:
    fps = get_fps(removed_df["CMILES"].tolist(), fptype=fp_type)
    fps_final = get_fps(final_df["CMILES"].tolist(), fptype=fp_type)
    for i, sim_type in enumerate(["Tanimoto", "Dice", "Cosine"]):
        rec_distances = [get_distance_vector(x, fps_final, sim_type) for x in fps]
        open(f"outputs/removed_distance_{sim_type}_{fp_type}.txt", "w").write(
            "\n".join([", ".join([str(y) for y in x]) for x in rec_distances])
        )
        logger.info(f"Saved 'outputs/removed_distance_{sim_type}_{fp_type}.txt'")
