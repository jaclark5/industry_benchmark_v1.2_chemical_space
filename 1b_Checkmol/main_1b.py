"""Use checkmol to analyze industry benchmark dataset



"""

import os
import sys
import json
from collections import defaultdict

from loguru import logger

from fg_assess_utilities import (
    plot_bar_chart,
    get_drug_categories,
    get_qca_ds,
    get_record_ids,
    get_functional_group_counts,
    get_occurrence_counts,
    draw_drug_patterns,
    plot_summary_stats,
)

sys.path.append("../5_greatest_common_structure")
from common_molecule_utilities import get_molecules, print_structures  # noqa: E402

os.makedirs("outputs", exist_ok=True)

ds = get_qca_ds("OpenFF Industry Benchmark Season 1 v1.1")
outlier_record_ids = get_record_ids(
    "https://raw.githubusercontent.com/openforcefield/sage-2.2.0/refs/heads/main/05_benchmark_forcefield/process_bm/problem_ids/all_r7_outliers.txt"
)
logger.info(
    f"There are {len(outlier_record_ids)} records that must be removed from Industry Benchmark v1.1 to make v1.2."
)

filename_ib = "outputs/functional_groups_ib.json"
filename_removed = "outputs/functional_groups_removed.json"

logger.info("Running Checkmol")
if not os.path.isfile(filename_ib):
    seen_cmiles = set()
    n_conformers = 0
    fg_breakdown_ib = {}
    fg_breakdown_removed = {}
    for i, (entry_name, spec_name, rec) in enumerate(ds.iterate_records()):
        if i % 1000 == 0:
            logger.info(f"Ran {i} of {ds.record_count}")
        if spec_name != "default":
            continue
        n_conformers += 1

        smiles = rec.initial_molecule.extras[
            "canonical_isomeric_explicit_hydrogen_mapped_smiles"
        ]
        if smiles in seen_cmiles:
            continue
        else:
            seen_cmiles.add(smiles)

        if rec.id in outlier_record_ids:
            fg_breakdown_removed[smiles] = get_functional_group_counts(smiles)
        else:
            fg_breakdown_ib[smiles] = get_functional_group_counts(smiles)

    logger.info(
        f"For conformers, there are {n_conformers} in v1.2 and {len(outlier_record_ids)} removed."
    )

    with open(filename_ib, "w") as f:
        json.dump(fg_breakdown_ib, f, indent=2)

    with open(filename_removed, "w") as f:
        json.dump(fg_breakdown_removed, f, indent=2)
else:
    with open(filename_ib, "r") as f:
        fg_breakdown_ib = json.load(f)

    with open(filename_removed, "r") as f:
        fg_breakdown_removed = json.load(f)

smiles_removed = set(fg_breakdown_removed.keys()) - set(fg_breakdown_ib.keys())
logger.info(
    f"For unique molecules, there are {len(fg_breakdown_ib)} in v1.2 and {len(smiles_removed)} removed from the dataset"
)

cutoff = 100
logger.info(
    f"Plotting functional groups that are present in more than {cutoff} molecules"
)
occurrence_counts, all_groups = get_occurrence_counts(
    fg_breakdown_ib,
    "outputs/occurence_nondrug.png",
    cutoff=cutoff,
)

#################################################################

logger.info("Import OE Molecules")
mols_ib = get_molecules(
    "../1_import_data_process_fps/outputs/industry_benchmark_v1.2_unique_mols.sdf"
)
mols_removed = get_molecules(
    "../1_import_data_process_fps/outputs/removed_records_unique_mols.sdf"
)


logger.info("Classify drug-like groups")
filename = "outputs/drug_types.png"
drug_categories: defaultdict[str, int] = defaultdict(lambda: 0)
nondrug_smiles = []
for smiles, mol in zip(fg_breakdown_ib.keys(), mols_ib):
    classifications = get_drug_categories(mol)
    if not classifications:
        nondrug_smiles.append((smiles, mol))
    else:
        for classification in classifications:
            drug_categories[classification] += 1
logger.info(
    f"There are {len(fg_breakdown_ib)-len(nondrug_smiles)} drug-like molecules and {len(nondrug_smiles)} that are not."
)

logger.info("Plotting drug-like molecule figures")
plot_bar_chart(drug_categories, filename)
draw_drug_patterns(list(drug_categories.keys()))

# for removed molecules
drug_categories_removed: defaultdict[str, int] = defaultdict(lambda: 0)
for smiles, mol in zip(fg_breakdown_removed.keys(), mols_removed):
    classifications = get_drug_categories(mol)
    for classification in classifications:
        drug_categories_removed[classification] += 1
logger.info(
    f"There are {sum(drug_categories_removed.values())} drug-like molecules out of {len(fg_breakdown_removed)} that were removed."
)
for classification, total in drug_categories_removed.items():
    logger.info(f"    {classification}: {total}")


logger.info(
    f"Plotting functional groups that are present in more than {cutoff} molecules that were not tagged as drug-like"
)
nondrug_fg_breakdown = {smiles: fg_breakdown_ib[smiles] for smiles, _ in nondrug_smiles}
occurrence_counts, all_groups = get_occurrence_counts(
    nondrug_fg_breakdown,
    "outputs/occurence_ib.png",
    cutoff=cutoff,
)


os.makedirs("outputs/remaining", exist_ok=True)
print_structures(
    [x[1] for x in nondrug_smiles], "outputs/remaining/remaining_structures.pdf"
)
print_structures(mols_removed, "outputs/remaining/removed_structures.pdf")

logger.info("Plotting summary statistics")
plot_summary_stats(
    {
        "Industry Benchmark v1.2": mols_ib,
        #        "Unclassified Structures": [x[1] for x in nondrug_smiles],
        #        "Removed Structures": mols_removed,
    },
    "outputs/summary_stats.png",
)
