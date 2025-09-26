"""Create dataframes for the Industry Benchmark v1.2 and Discarded molecules from v1.1"""

from urllib.request import urlopen
from loguru import logger

import pandas as pd

from qcportal import PortalClient

from fp_utilities import get_fps

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

final_df = final_df.groupby("CMILES", as_index=False).agg({"Name": lambda x: list(x)})
final_df.rename(columns={"Name": "Names"}, inplace=True)
logger.info(
    f"For unique molecules, there are {len(final_df)} in v1.2 and {len(removed_df)} removed."
)


# Calculate OE fingerprints for the molecules:
final_df["Circular FP"] = get_fps(final_df["CMILES"].tolist(), fptype="Circular")
final_df["MACCS FP"] = get_fps(final_df["CMILES"].tolist(), fptype="MACCS")
final_df["LINGO FP"] = get_fps(final_df["CMILES"].tolist(), fptype="LINGO")

removed_df["Circular FP"] = get_fps(removed_df["CMILES"].tolist(), fptype="Circular")
removed_df["MACCS FP"] = get_fps(removed_df["CMILES"].tolist(), fptype="MACCS")
removed_df["LINGO FP"] = get_fps(removed_df["CMILES"].tolist(), fptype="LINGO")
logger.info("Added Circular, MACCS, and LINGO fingerprints for each CMILES string.")

removed_df.to_csv("removed_records_unique_fps.csv", index=False)
logger.info("Saved consolidated fingerprints 'removed_record_unique_fps.csv'")
final_df.to_csv("industry_benchmark_v1.2_unique_fps.csv", index=False)
logger.info("Saved consolidated fingerprints 'industry_benchmark_v1.2_unique_fps.csv'")
