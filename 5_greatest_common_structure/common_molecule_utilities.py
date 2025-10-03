from loguru import logger

from openeye import oechem
from openeye import oedepict
from getcore import GetCoreFragment


def get_molecules(filename: str) -> list[oechem.OEMol]:
    """
    Extract a list of OpenEye Molecules from an SDF file.

    Parameters
    ----------
    filename : str
        Path to the SDF file containing molecules.

    Returns
    -------
    list[oechem.OEMol]
        List of OpenEye molecules.
    """
    ifs = oechem.oemolistream()
    if not ifs.open(filename):
        oechem.OEThrow.Fatal(f"Unable to open {filename} for reading")
    if ifs.GetFormat() != oechem.OEFormat_SDF:
        oechem.OEThrow.Fatal(f"{filename} input file has to be an SDF file")
    logger.info(f"Connected to SDF {filename}")

    logger.info("Get Molecules")
    mols = []
    for mol in ifs.GetOEGraphMols():
        mols.append(oechem.OEGraphMol(mol))

    return mols


def get_cluster_ref_mols(
    cluster_dict: dict[int, list[oechem.OEMol]], filename: str
) -> list[oechem.OEMol]:
    """
    Generate reference molecules for each cluster and save a summary to a file.

    Parameters
    ----------
    cluster_dict : dict[int, list[oechem.OEMol]]
        Dictionary mapping cluster IDs to lists of molecules.
    filename : str
        Path to save the summary file.

    Returns
    -------
    list[oechem.OEMol]
        List of reference molecules for each cluster.
    """
    summary = []
    ref_mols = []
    for cluster_id, tmp_mols in cluster_dict.items():
        ref_mol = GetCoreFragment(tmp_mols, 1, 18)
        logger.info(f"Obtained reference molecule for cluster of size: {len(tmp_mols)}")
        if ref_mol is not None:
            oechem.OESetSDData(ref_mol, "CLUSTER_ID", str(cluster_id))
            oechem.OESetSDData(ref_mol, "N_CLUSTER_MOLS", str(len(tmp_mols)))
        summary.append(
            [
                cluster_id,
                len(tmp_mols),
                oechem.OECreateSmiString(ref_mol) if ref_mol is not None else None,
            ]
        )
        ref_mols.append(ref_mol)

    logger.info("Saving summary...")
    with open(filename, "w") as f:
        f.write("Cluster ID, N Cluster Molecules, SMILES\n")
        for cluster_id, n_mols, smiles in summary:
            f.write(f"{cluster_id}, {n_mols}, {smiles}\n")

    return ref_mols


def print_structures(
    mols: list[oechem.OEMol], filename: str, cols: int = 4, rows: int = 5
) -> None:
    """
    Generate a grid of molecular structures and save to a multi-page PDF file.

    Parameters
    ----------
    mols : list[oechem.OEMol]
        List of OpenEye molecules to depict.
    filename : str
        Path to save the generated PDF file.
    cols : int, optional
        Number of columns in a page containing 2D depictions of molecules.
        Defaults to 4.
    rows : int, optional
        Number of rows in a page containing 2D depictions of molecules.
        Defaults to 5.

    Returns
    -------
    None
    """

    opts = oedepict.OE2DMolDisplayOptions()
    opts.SetDimensions(150, 150, oedepict.OEScale_AutoScale)

    report = oedepict.OEReport(oedepict.OEReportOptions(rows, cols))

    for i, mol in enumerate(mols):
        if mol is None:
            continue

        mol.SetTitle(f"Molecule {i}")
        oedepict.OEPrepareDepiction(mol)

        cell = report.NewCell()
        disp = oedepict.OE2DMolDisplay(mol, opts)
        oedepict.OERenderMolecule(cell, disp)

    oedepict.OEWriteReport(filename, report)
