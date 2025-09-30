"""Distance matrices using various similarity methods with available fingerprints.

The major inputs of this module are:

- ../1_import_data_process_fps/*.sdf
    SDF files of Industry Benchmark v1.2 and those that were removed from v1.1 containing OpenEye fingerprints

The major outputs of this module are in the outputs directory:

- outputs/distance_distributions.png
    Plotted distributions of similarity values based on method and fingerprint input.
- outputs/distances_{Tanimoto | Dice | Cosine}_{Circular | MACCS | Lingo}.txt files
    Containing the full distance matrix (1 - similarity) for "OpenFF Industry Benchmark Season 1 v1.2"
- outputs/removed_distance_{Tanimoto | Dice | Cosine}_{Circular | MACCS | Lingo}.txt files
    Each row contains the distance between that molecule and all molecules in the Industry Benchmark v1.2.

"""

import os

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np

from openeye import oechem
from openeye import oegraphsim
from openeye.oegraphsim import OEFingerPrint

from distance_utilities import get_distance_matrix, get_distance_vector

os.makedirs("outputs", exist_ok=True)


logger.info(
    "_______________________________________________________________________________________________________________"
)
logger.info(
    "__________________________ Get Distances between Molecules in Industry Benchmark v1.2 __________________________"
)

# SDF for the cleaned Industry Dataset v1.2 and one for the removed entries:
filename_indbench = (
    "../1_import_data_process_fps/outputs/industry_benchmark_v1.2_unique_mols.sdf"
)
ifs_indbench = oechem.oemolistream()
if not ifs_indbench.open(filename_indbench):
    oechem.OEThrow.Fatal(f"Unable to open {filename_indbench} for reading")
if ifs_indbench.GetFormat() != oechem.OEFormat_SDF:
    oechem.OEThrow.Fatal(f"{filename_indbench} input file has to be an SDF file")
logger.info(f"Connected to SDF {filename_indbench}")

logger.info("Get Fingerprints")
fps_ib: dict[str, list[OEFingerPrint]] = {x: [] for x in ["Circular", "MACCS", "Lingo"]}
for mol in ifs_indbench.GetOEGraphMols():
    for dp in oechem.OEGetSDDataPairs(mol):
        if oegraphsim.OEIsValidFPTypeString(dp.GetTag()):
            fptypestr = dp.GetTag()
            fphexdata = dp.GetValue()
            fp = oegraphsim.OEFingerPrint()
            fptype = oegraphsim.OEGetFPType(fptypestr)
            fp.SetFPTypeBase(fptype)
            fp.FromHexString(fphexdata)

            if "Circular" in fptypestr:
                fps_ib["Circular"].append(fp)
            elif "MACCS" in fptypestr:
                fps_ib["MACCS"].append(fp)
            elif "Lingo" in fptypestr:
                fps_ib["Lingo"].append(fp)
logger.info("Imported fingerprints, Circular, MACCS, and Lingo")

fig, axs = plt.subplots(1, 3, figsize=(8, 5), sharey=True)
similarity_stats = []
for j, fp_type in enumerate(["Lingo", "Circular", "MACCS"]):
    if len(fps_ib[fp_type]) == 0:
        raise ValueError(f"Fingerprint {fp_type} was not found in SDF files.")
    for i, sim_type in enumerate(["Tanimoto", "Dice", "Cosine"]):
        logger.info(f"Getting {sim_type} similarity distances")
        filename = f"outputs/distances_{sim_type}_{fp_type}.txt"
        if not os.path.isfile(filename):
            distances = get_distance_matrix(fps_ib[fp_type], sim_type)
            logger.info(f"Saving 'outputs/distances_{sim_type}_{fp_type}.txt'")
            np.savetxt(filename, distances, delimiter=",")
        #            with open(filename, "w") as f:
        #                f.write(f"{distances[1]}\n")
        #                for n in distances[0]:
        #                    f.write(f"{n}\n")
        else:
            logger.info(f"Imported 'outputs/distances_{sim_type}_{fp_type}.txt'")
            #            data = [
            #                float(x.strip())
            #                for x in open(filename).read().strip().split("\n")
            #                if x != ""
            #            ]
            #            distances = (data[1:], int(data[0]))
            distances = np.genfromtxt(filename, delimiter=",")
        # Plot similarity distribution
        upper_triangle_distances = distances[np.triu_indices_from(distances, k=1)]
        similarity_stats.append(
            [
                f"{sim_type} {fp_type}",
                np.mean(upper_triangle_distances),
                np.std(upper_triangle_distances),
            ]
        )
        label = rf"{fp_type}: $\mu$ = {similarity_stats[-1][1]:0.2f} $\sigma$ = {similarity_stats[-1][2]:0.2f}"
        axs[i].hist(
            upper_triangle_distances,
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
with open("outputs/similarity_stats.csv", "w") as f:
    f.write("Similarity_FP, Mean Distance, Std Distance\n")
    for line in similarity_stats:
        f.write(", ".join([str(x) for x in line]) + "\n")

###############################################################################################################################

logger.info(
    "_______________________________________________________________________________________________________________"
)
logger.info(
    "________________ Get Distances between Removed Molecules with those in Industry Benchmark v1.2 ________________"
)

filename_removed = (
    "../1_import_data_process_fps/outputs/removed_records_unique_mols.sdf"
)
ifs_removed = oechem.oemolistream()
if not ifs_removed.open(filename_removed):
    oechem.OEThrow.Fatal(f"Unable to open {filename_removed} for reading")
if ifs_removed.GetFormat() != oechem.OEFormat_SDF:
    oechem.OEThrow.Fatal(f"{filename_removed} input file has to be an SDF file")
logger.info(f"Connected to SDF {filename_removed}")

fps_removed: dict[str, list[OEFingerPrint]] = {"Circular": [], "MACCS": [], "Lingo": []}
for mol in ifs_removed.GetOEGraphMols():
    for dp in oechem.OEGetSDDataPairs(mol):
        if oegraphsim.OEIsValidFPTypeString(dp.GetTag()):
            fptypestr = dp.GetTag()
            fphexdata = dp.GetValue()
            fp = oegraphsim.OEFingerPrint()
            fptype = oegraphsim.OEGetFPType(fptypestr)
            fp.SetFPTypeBase(fptype)
            fp.FromHexString(fphexdata)

            if "Circular" in fptypestr:
                fps_removed["Circular"].append(fp)
            elif "MACCS" in fptypestr:
                fps_removed["MACCS"].append(fp)
            elif "Lingo" in fptypestr:
                fps_removed["Lingo"].append(fp)

for fp_type in ["Circular", "MACCS", "Lingo"]:
    for i, sim_type in enumerate(["Tanimoto", "Dice", "Cosine"]):
        filename = f"outputs/removed_distance_{sim_type}_{fp_type}.txt"
        open(filename, "w").write(
            "#distances for each removed molecule with respect to industry benchmark v1.2\n"
        )
        for fp in fps_removed[fp_type]:
            rec_distances = get_distance_vector(fp, fps_ib[fp_type], sim_type)
            open(filename, "a").write(", ".join([str(y) for y in rec_distances]) + "\n")
        logger.info(f"Saved 'outputs/removed_distance_{sim_type}_{fp_type}.txt'")
