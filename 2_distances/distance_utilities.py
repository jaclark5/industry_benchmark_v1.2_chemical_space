"""Support functions to calculate molecular fingerprints using OpenEye Toolkits"""

import numpy as np
from openeye import oechem
from openeye import oegraphsim
from openeye.oegraphsim import OEFingerPrint

# Suppress OpenEye warnings
oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)


def get_distance_upper_triangle(
    fps: list[OEFingerPrint], metric: str = "Tanimoto"
) -> tuple[list[float], int]:
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
    tuple[list[float], int]
        A tuple containing:
        - The upper triangle distance matrix where the distance is ``1 - similarity``.
        - The number of points (fingerprints) in the input list.

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
    distance_data = []
    for i in range(n):
        for j in range(i):
            sim = metric_dict[metric](fps[i], fps[j])
            distance_data.append(1 - sim)
    return distance_data, n


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
