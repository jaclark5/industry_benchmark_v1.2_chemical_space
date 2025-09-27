"""Support functions to calculate molecular fingerprints using OpenEye Toolkits"""

from typing import List, Tuple
from openeye import oechem
from openeye import oegraphsim
from openeye.oegraphsim import OEFingerPrint

# Suppress OpenEye warnings
oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)


def get_fps(smiles_list: List[str], fptype: str = "Circular") -> List[OEFingerPrint]:
    """Compute OpenEye fingerprints for a list of molecules.

    Parameters
    ----------
    smiles_list : list
        List of SMILES strings
    fptype : str, optional, default="Circular2"
        OpenEye fingerprint type, "Circular", "MACCS", or "LINGO"

    Returns
    -------
    List[OEFingerPrint]
        List of OpenEye fingerprints
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


def get_distance_matrix(
    fps: List[OEFingerPrint], metric: str = "Tanimoto"
) -> Tuple[List[float], int]:
    """
    Generate a distance matrix (1 - similarity) for a list of fingerprints using the specified similarity metric.

    Parameters
    ----------
    fps : list
        List of OpenEye fingerprints
    metric : str, optional, default="Tanimoto"
        Similarity metric to use, one of "Tanimoto", "Dice", or "Cosine"

    Returns
    -------
    distance_data : List[float]
        The upper triangle distance matrix where the distance is ``1 - similarity``
    n : int
        The number of points (fingerprints) in the input list
    """
    metric_dict = {
        "Tanimoto": oegraphsim.OETanimoto,
        "Dice": oegraphsim.OEDice,
        "Cosine": oegraphsim.OECosine,
    }
    if metric not in metric_dict:
        raise ValueError(
            f"Unsupported metric '{metric}'. Choose from {list(metric_dict.keys())}."
        )

    n = len(fps)
    distance_data = []
    for i in range(n):
        for j in range(i):
            sim = metric_dict[metric](fps[i], fps[j])
            distance_data.append(1 - sim)
    return distance_data, n


def get_distance_vector(
    fp: OEFingerPrint, fps: List[OEFingerPrint], metric: str = "Tanimoto"
) -> List[float]:
    """
    Generate a distance vector (1 - similarity) for a fingerprint against a list of fingerprints using the specified similarity metric.
    Supported metrics: "Tanimoto", "Dice", "Cosine"

    Parameters
    ----------
    fp : OEFingerPrint
        OpenEye fingerprint
    fps : list[OEFingerPrint]
        List of OpenEye fingerprints
    metric : str, optional, default="Tanimoto"
        Similarity metric to use, one of "Tanimoto", "Dice", or "Cosine"

    Returns
    -------
    List[float]
        Distance vector of the provided fingerprint with other fingerprints
    """
    metric_dict = {
        "Tanimoto": oegraphsim.OETanimoto,
        "Dice": oegraphsim.OEDice,
        "Cosine": oegraphsim.OECosine,
    }
    if metric not in metric_dict:
        raise ValueError(
            f"Unsupported metric '{metric}'. Choose from {list(metric_dict.keys())}."
        )

    return [1 - metric_dict[metric](fp, x) for x in fps]
