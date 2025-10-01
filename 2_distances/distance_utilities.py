"""Support functions to calculate molecular fingerprints using OpenEye Toolkits"""

import numpy as np
from loguru import logger

from openeye import oechem
from openeye import oegraphsim
from openeye.oegraphsim import OEFingerPrint

# Suppress OpenEye warnings
oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)


def get_fingerprints(
    filename: str, fp_types: list[str] = ["Circular", "MACCS", "Lingo"]
) -> dict[str, list[OEFingerPrint]]:
    """
    Extract molecular fingerprints from an SDF file using OpenEye Toolkits.

    Parameters
    ----------
    filename : str
        Path to the SDF file containing molecules and fingerprint data.
    fp_types : list[str], optional
        List of fingerprint types to extract (default: ["Circular", "MACCS", "Lingo"]).

    Returns
    -------
    dict[str, list[OEFingerPrint]]
        Dictionary mapping fingerprint type to list of OEFingerPrint objects.
    """

    ifs = oechem.oemolistream()
    if not ifs.open(filename):
        oechem.OEThrow.Fatal(f"Unable to open {filename} for reading")
    if ifs.GetFormat() != oechem.OEFormat_SDF:
        oechem.OEThrow.Fatal(f"{filename} input file has to be an SDF file")
    logger.info(f"Connected to SDF {filename}")

    logger.info("Get Fingerprints")
    fps: dict[str, list[OEFingerPrint]] = {x: [] for x in fp_types}
    for mol in ifs.GetOEGraphMols():
        for dp in oechem.OEGetSDDataPairs(mol):
            if oegraphsim.OEIsValidFPTypeString(dp.GetTag()):
                fptypestr = dp.GetTag()
                fphexdata = dp.GetValue()
                fp = oegraphsim.OEFingerPrint()
                fptype = oegraphsim.OEGetFPType(fptypestr)
                fp.SetFPTypeBase(fptype)
                fp.FromHexString(fphexdata)

                for fp_type in fp_types:
                    if fp_type in fptypestr:
                        fps[fp_type].append(fp)

    logger.info(f"Imported fingerprints, {', '.join(fp_types)}")
    return fps


def get_distance_matrix(
    fps: list[OEFingerPrint], metric: str = "Tanimoto"
) -> np.ndarray:
    """
    Generate a distance matrix (1 - similarity) for a list of fingerprints using the specified similarity metric.

    Parameters
    ----------
    fps : list[OEFingerPrint]
        List of OpenEye fingerprints.
    metric : str, optional
        Similarity metric to use. Defaults to "Tanimoto".

    Returns
    -------
    np.ndarray
        Full matrix ``1 - similarity``.

    Raises
    ------
    ValueError
        If an unsupported metric is provided.
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
    distance_data = np.zeros((n, n))

    for i in range(n):
        distance_data[i, :] = np.array(
            [1 - metric_dict[metric](fps[i], fps[j]) for j in range(n)]
        )

    return distance_data.astype(np.float32)


def get_distance_vector(
    fp: OEFingerPrint, fps: list[OEFingerPrint], metric: str = "Tanimoto"
) -> np.ndarray:
    """
    Generate a distance vector (1 - similarity) for a fingerprint against a list of fingerprints using the specified similarity metric.

    Parameters
    ----------
    fp : OEFingerPrint
        OpenEye fingerprint.
    fps : list[OEFingerPrint]
        List of OpenEye fingerprints.
    metric : str, optional
        Similarity metric to use. Defaults to "Tanimoto".

    Returns
    -------
    np.ndarray
        Distance vector of the provided fingerprint with other fingerprints as a float32 NumPy array.

    Raises
    ------
    ValueError
        If an unsupported metric is provided.
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

    return np.array([1 - metric_dict[metric](fp, x) for x in fps], dtype=np.float32)
