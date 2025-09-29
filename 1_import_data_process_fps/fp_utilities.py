"""Support functions to calculate molecular fingerprints using OpenEye Toolkits"""

from openeye import oechem
from openeye import oegraphsim
from openeye.oechem import OEMolBase

# Suppress OpenEye warnings
oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)


def get_mol_fps(
    smiles: str, fptypes: list[str] = ["Circular", "MACCS", "LINGO"]
) -> OEMolBase:
    """Compute OpenEye fingerprints for a molecule.

    Parameters
    ----------
    smiles : str
        A SMILES string representing the molecule.
    fptypes : list[str], optional
        A list of OpenEye fingerprint types. Supported types are "Circular", "MACCS", and "LINGO".
        Defaults to ["Circular", "MACCS", "LINGO"].

    Returns
    -------
    OEMolBase
        An OpenEye molecule object with fingerprints stored as data.

    Raises
    ------
    ValueError
        If an unsupported fingerprint type is provided in `fptypes`.
    """
    fptype_dict = {
        "Circular": oegraphsim.OEMakeCircularFP,
        "MACCS": oegraphsim.OEMakeMACCS166FP,
        "LINGO": oegraphsim.OEMakeLingoFP,
    }

    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, smiles)
    for fptype in fptypes:
        if fptype not in fptype_dict:
            raise ValueError(
                f"Unsupported metric '{fptype}'. Choose from {list(fptype_dict.keys())}."
            )

        fp = oegraphsim.OEFingerPrint()
        fptype_dict[fptype](fp, mol)
        fptypestr = fp.GetFPTypeBase().GetFPTypeString()
        fphexdata = fp.ToHexString()
        oechem.OESetSDData(mol, fptypestr, fphexdata)

    return mol
