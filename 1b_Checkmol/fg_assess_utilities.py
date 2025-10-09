"""Support functions to calculate molecular fingerprints using OpenEye Toolkits"""

import json
import os
from urllib.request import urlopen

import matplotlib.pyplot as plt
import matplotlib as mpl
from loguru import logger
import numpy as np
from openeye import oechem, oedepict

from qcportal import PortalClient

from yammbs.checkmol import analyze_functional_groups

mpl.rcParams["font.family"] = "PT Serif"
label_size = 10
tick_size = 8
rotate = 90
ha = "center"


def get_qca_ds(ds_name: str, ds_type: str = "optimization"):
    """
    Retrieve a dataset from QCArchive.

    Parameters
    ----------
    ds_name : str
        The name of the dataset to retrieve.
    ds_type : str, optional
        The type of the dataset, by default "optimization".

    Returns
    -------
    PortalClient.Dataset
        The requested dataset object.
    """
    qc_client = PortalClient("https://api.qcarchive.molssi.org:443/", cache_dir=".")
    ds = qc_client.get_dataset(ds_type, ds_name)
    logger.info(f"Pulled optimization dataset '{ds_name}' from QCArvive")

    return ds


def get_record_ids(url: str) -> set[int]:
    """
    Retrieve record IDs from a given URL.

    Parameters
    ----------
    url : str
        The URL to fetch record IDs from.

    Returns
    -------
    set[int]
        A set of record IDs.
    """
    with urlopen(url) as response:
        file_content = response.read().decode()
    return {int(x) for x in file_content.splitlines()}


def get_functional_group_counts(smiles: str) -> dict[str, int]:
    """
    Analyze functional groups in a molecule and return their counts.

    Parameters
    ----------
    smiles : str
        The SMILES representation of the molecule.

    Returns
    -------
    dict[str, int]
        A dictionary mapping functional group names to their counts.
    """
    groups_and_counts: dict[str, int] = {
        gp.value: count for gp, count in analyze_functional_groups(smiles).items()
    }
    if "Hydroxy" in groups_and_counts:
        del groups_and_counts["Hydroxy"]
    return groups_and_counts


def get_histogram(
    fg_breakdown: dict[str, dict[str, int]], filename: str, cutoff: int = 0
) -> tuple[np.ndarray, list[str]]:
    """
    Generate a 2D histogram of functional group co-occurrences.

    Parameters
    ----------
    fg_breakdown : dict[str, dict[str, int]]
        A dictionary where keys are molecule identifiers and values are dictionaries
        of functional group counts.
    filename : str
        The filename to save the histogram image.
    cutoff : int, optional
        If all values in a row/column have less than this number, then the row/column is removed. Default is 0.

    Returns
    -------
    cooccurrence_matrix : np.ndarray
        A 2D numpy array representing the co-occurrence matrix of functional groups.
    all_groups : list[str]
        A list of all unique functional groups in the dataset.
    """
    # Get all unique functional groups
    tmp_all_groups: set[str] = set()
    for molecule_data in fg_breakdown.values():
        tmp_all_groups.update(molecule_data.keys())
    all_groups = sorted(list(tmp_all_groups))

    # Create matrix to count co-occurrences
    occurrence_array = np.zeros(len(all_groups))
    for molecule_data in fg_breakdown.values():
        present_groups = [group for group, count in molecule_data.items() if count > 0]
        for group1 in present_groups:
            occurrence_array[all_groups.index(group1)] += 1

    # Remove rows/columns with all values below cutoff
    to_keep = occurrence_array >= cutoff
    occurrence_array = occurrence_array[to_keep]
    all_groups = [group for i, group in enumerate(all_groups) if to_keep[i]]
    n_groups = len(all_groups)

    # Sort occurrence_array and all_groups in descending order of occurrence_array
    sorted_indices = np.argsort(-occurrence_array)
    occurrence_array = occurrence_array[sorted_indices]
    all_groups = [all_groups[i] for i in sorted_indices]

    # Create the heatmap
    plt.figure(figsize=(6, 6))
    plt.bar(
        range(n_groups),
        occurrence_array / len(fg_breakdown),
        color=(0.007843137, 0.333333333, 0.505882353),
    )
    plt.xlim((-0.6, n_groups - 0.4))
    plt.ylabel("Fraction of Structures", fontsize=label_size)
    plt.tick_params(axis="y", labelsize=tick_size)
    plt.xticks(range(n_groups), all_groups, rotation=rotate, ha=ha, fontsize=tick_size)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()

    return occurrence_array, all_groups


def _get_drug_patterns() -> dict[str, tuple[str, oechem.OEQMol]]:
    drug_smiles = json.load(open("reference/drug_backbones.json", "r"))
    output = {}
    for name, smiles in drug_smiles.items():
        patt = oechem.OEQMol()
        oechem.OEParseSmarts(patt, smiles)
        output[name] = (smiles, patt)
    return output


DRUG_PATTERNS = _get_drug_patterns()


def get_drug_categories(mol: oechem.OEQMol) -> list:
    """
    Check the drug category of a molecule based on its SMILES representation.

    Parameters
    ----------
    mol : oechem.OEQMol
        OpenEye Molecule

    Returns
    -------
    list
        The names of the drug categories if a match is found.
    """

    patterns = []
    for name, (_, mol_patt) in DRUG_PATTERNS.items():
        ss = oechem.OESubSearch(mol_patt)
        if ss.SingleMatch(mol):
            patterns.append(name)
    return patterns


def plot_bar_chart(dictionary: dict[str, int], filename: str, cutoff: int = 5) -> None:
    """
    Plot a bar chart from a dictionary and save it to a file.

    Parameters
    ----------
    dictionary : dict[str, int]
        A dictionary where keys are labels and values are counts.
    filename : str
        The filename to save the bar chart image.
    cutoff : int, optional
        Number of molecules in this class that must be present to plot

    Returns
    -------
    None
    """
    # Filter dictionary to only include items with values greater than cutoff
    dictionary = {k: v for k, v in dictionary.items() if v > cutoff}
    if not dictionary:
        logger.error("No data was provided.")
        return
    plt.figure(figsize=(6, 4))
    plt.bar(
        list(dictionary.keys()),
        list(dictionary.values()),
        color=(0.007843137, 0.333333333, 0.505882353),
    )  # Convert to lists for compatibility
    for i, (key, value) in enumerate(dictionary.items()):
        plt.text(
            i,
            value + 0.01 * max(dictionary.values()),
            str(value),
            ha="center",
            va="bottom",
        )
    plt.ylim(0, max(dictionary.values()) * 1.1)
    plt.ylabel("Count", fontsize=label_size)
    plt.xticks(rotation=rotate, ha=ha, fontsize=tick_size)
    plt.yticks(fontsize=label_size)  # NoteHere
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()


def draw_drug_patterns(keys: list[str], output_dir: str = "outputs/structures") -> None:
    """
    Draw 2D representations of backbone SMIRKS patterns with connection points.

    Parameters
    ----------
    keys : list[str]
        List of keys from DRUG_PATTERNS dictionary to draw structures for.
    output_dir : str, optional
        Directory to save the structure images, by default "outputs".

    Returns
    -------
    None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for key in keys:
        if key not in DRUG_PATTERNS:
            logger.warning(f"Key '{key}' not found in DRUG_PATTERNS dictionary")
            continue

        _, mol = DRUG_PATTERNS[key]
        oedepict.OEPrepareDepiction(mol)
        output_file = os.path.join(output_dir, f"{key}.png")
        oedepict.OERenderMolecule(output_file, mol)

        logger.info(f"Saved structure for '{key}' to {output_file}")


def plot_summary_stats(
    mol_dict: dict[str, list[oechem.OEQMol]], filename: str = "summary_stats.png"
) -> None:
    """
    Calculate and plot summary statistics for different OpenEye molecule datasets.

    Parameters
    ----------
    mol_dict : dict[str, list[oechem.OEQMol]]
        Dictionary where keys are dataset names and values are lists of SMILES strings.
    filename : str, optional
        Filename to save the plot, by default "summary_stats.png".

    Returns
    -------
    None
    """
    base_categories = [
        "6-Member\nNonaromatic Rings",
        "6-Member\nAromatic Rings",
        #        "Fused 5 and 6-Member\nNonaromatic Rings",
        #        "Fused 5 and 6-Member\nAromatic Rings",
        #        "2 Fused 6-Member\nNonaromatic Rings",
        #        "2 Fused 6-Member\nAromatic Rings",
        "5-Member Rings",
    ]

    # First pass: collect all unique elements that are not H, C, S, O, N
    allowed_elements = {"H", "C"}
    all_other_elements = set()
    charges = set()
    molecular_weights = []

    for mol_list in mol_dict.values():
        for mol in mol_list:
            mol_elements = {
                oechem.OEGetAtomicSymbol(atom.GetAtomicNum()) for atom in mol.GetAtoms()
            }
            other_elements = mol_elements - allowed_elements
            all_other_elements.update(other_elements)
            charges.add(sum(a.GetFormalCharge() for a in mol.GetAtoms()))

    # Create categories list with other elements
    categories = (
        base_categories
        + [f"Contains {elem}" for elem in sorted(all_other_elements)]
        + [f"{chg} Charge" for chg in sorted(charges)]
    )
    results: dict[str, dict[str, float]] = {category: {} for category in categories}

    for dataset_name, mol_list in mol_dict.items():
        total_mols = len(mol_list)
        if total_mols == 0:
            continue

        six_ring_count = 0
        six_aromatic_count = 0
        five_ring_count = 0
        element_counts = {elem: 0 for elem in all_other_elements}
        charge_counts = {chg: 0 for chg in sorted(charges)}

        tmp_mol_weight = []
        for mol in mol_list:
            # Check for 6-member rings
            n_rings, ring_idx_per_atom = oechem.OEDetermineRingSystems(mol)
            if n_rings > 0:
                found = [False for _ in range(7)]
                for ii in range(1, n_rings + 1):
                    ring_atom_idx = [
                        atom_idx
                        for atom_idx, ring_id in enumerate(ring_idx_per_atom)
                        if ring_id == ii
                    ]
                    ring_atoms = [
                        atom
                        for atom in mol.GetAtoms()
                        if atom.GetIdx() in ring_atom_idx
                    ]
                    is_aromatic = any([atom.IsAromatic() for atom in ring_atoms])
                    ring_size = len(ring_atoms)
                    if ring_size == 6:
                        # Check if 6-member ring is aromatic
                        if is_aromatic and not found[0]:
                            found[0] = True
                            six_aromatic_count += 1
                        elif not is_aromatic and not found[1]:
                            found[1] = True
                            six_ring_count += 1
                    elif ring_size == 5 and not found[6]:
                        found[6] = True
                        five_ring_count += 1

            # Check for specific other elements
            mol_elements = {
                oechem.OEGetAtomicSymbol(atom.GetAtomicNum()) for atom in mol.GetAtoms()
            }
            other_elements = mol_elements - allowed_elements
            for elem in other_elements:
                if elem in element_counts:
                    element_counts[elem] += 1
            charge_counts[sum(a.GetFormalCharge() for a in mol.GetAtoms())] += 1
            tmp_mol_weight.append(
                sum(
                    oechem.OEGetAverageWeight(atom.GetAtomicNum())
                    for atom in mol.GetAtoms()
                )
            )

        molecular_weights.append(tmp_mol_weight)
        # Store fractions
        results["6-Member\nNonaromatic Rings"][dataset_name] = (
            six_ring_count / total_mols
        )
        results["6-Member\nAromatic Rings"][dataset_name] = (
            six_aromatic_count / total_mols
        )
        results["5-Member Rings"][dataset_name] = five_ring_count / total_mols
        #        results["Fused 5 and 6-Member\nNonaromatic Rings"][dataset_name] = nine_ring / total_mols
        #        results["Fused 5 and 6-Member\nAromatic Rings"][dataset_name] = nine_aro / total_mols
        #        results["2 Fused 6-Member\nNonaromatic Rings"][dataset_name] = ten_ring / total_mols
        #        results["2 Fused 6-Member\nAromatic Rings"][dataset_name] = ten_aro / total_mols

        for elem in all_other_elements:
            results[f"Contains {elem}"][dataset_name] = (
                element_counts[elem] / total_mols
            )
        for chg in sorted(charges):
            results[f"{chg} Charge"][dataset_name] = charge_counts[chg] / total_mols

    # Create grouped bar chart
    dataset_names = list(mol_dict.keys())
    n_datasets = len(dataset_names)
    n_categories = len(categories)

    fig, axs = plt.subplots(
        1, 2, figsize=(6, 4), gridspec_kw={"width_ratios": [0.75, 0.25]}
    )

    x = np.arange(n_categories)
    width = 0.8 / n_datasets
    alpha = 0.5 if len(dataset_names) > 1 else 1

    for i, dataset_name in enumerate(dataset_names):
        values = [results[cat].get(dataset_name, 0) for cat in categories]
        offset = (i - n_datasets / 2 + 0.5) * width
        bars = axs[0].bar(
            x + offset,
            values,
            width,
            label=dataset_name,
            color=(0.007843137, 0.333333333, 0.505882353),
        )

        # Add fraction values above bars
        for bar, value in zip(bars, values):
            axs[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )

        axs[1].hist(
            molecular_weights[i],
            bins=100,
            alpha=alpha,
            color=(0.007843137, 0.333333333, 0.505882353),
            label=dataset_name,
            density=True,
        )

    axs[0].set_ylabel("Fraction of Structures", fontsize=label_size)
    axs[0].set_xticks(x)
    axs[0].set_xlim((x[0] - 0.6, x[-1] + 0.6))
    axs[0].set_xticklabels(categories, rotation=rotate, ha=ha, fontsize=tick_size)
    if len(dataset_names) > 1:
        axs[0].legend()
    axs[0].set_ylim(0, 1.0)
    axs[0].tick_params(axis="y", labelsize=tick_size)

    axs[1].set_xlabel("MW [Da]", fontsize=label_size)
    all_molecular_weight_extreme = [
        min(min(weights) for weights in molecular_weights),
        max(max(weights) for weights in molecular_weights),
    ]
    xticks = range(0, int(all_molecular_weight_extreme[1]) + 100, 100)
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xticks, rotation=rotate, fontsize=tick_size)
    axs[1].set_xlim(
        (all_molecular_weight_extreme[0], all_molecular_weight_extreme[1] / 2)
    )
    logger.info(
        f"MW min: {all_molecular_weight_extreme[0]}, max: {all_molecular_weight_extreme[1]}"
    )
    axs[1].set_yticks([])

    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()

    logger.info(f"Saved summary statistics plot to {filename}")
