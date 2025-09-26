"""Support functions to calculate molecular fingerprints using OpenEye Toolkits"""

from openeye import oechem
from openeye import oegraphsim


def get_fps(smiles_list: list, fptype: str = "Circular"):
    """Compute OpenEye fingerprints for a list of molecules.

    Parameters
    ----------
    smiles_list : list
        List of SMILES strings
    fptype : str, optional, default="Circular2"
        OpenEye fingerprint type, "Circular", "MACCS", or "LINGO"
    """
    fptype_dict = {
        "Circular": oegraphsim.OEMakeCircularFP,
        "MACCS": oegraphsim.OEMakeMACCS166FP,
        "LINGO": oegraphsim.OEMakeLingoFP,
    }
    if fptype not in fptype_dict:
        raise ValueError(
            f"Unsupported metric '{fptype}'. Choose from {list(fptype_dict.keys())}."
        )

    fps = []
    for smiles in smiles_list:
        mol = oechem.OEGraphMol()
        oechem.OESmilesToMol(mol, smiles)

        fp = oegraphsim.OEFingerPrint()
        fptype_dict[fptype](fp, mol)
        fps.append(fp)

    return fps
