"""Create SDF files for the Industry Benchmark v1.2 and discarded molecules from v1.1 that contain:

- *: OE Molecule information
- NAME: QCArchive entry name
- CMILES: Canonical SMILES from QCArchive used to create the molecule
- Circular: Circular molecule fingerprint
- MACCS: MACCS molecule fingerprint
- LINGO: LINGO molecule fingerprint

"""

import os
from urllib.request import urlopen

from loguru import logger
from qcportal import PortalClient

from fp_utilities import get_mol_fps
from openeye import oechem

os.makedirs("outputs", exist_ok=True)

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

# Create SDF for the cleaned Industry Dataset v1.2 and one for the removed entries:
filename_indbench = "outputs/industry_benchmark_v1.2_unique_mols.sdf"
filename_removed = "outputs/removed_records_unique_mols.sdf"

if not os.path.isfile(filename_indbench) or not os.path.isfile(filename_removed):
    # Prepare output SDFs
    ofs_indbench = oechem.oemolostream()
    if not ofs_indbench.open(filename_indbench):
        oechem.OEThrow.Fatal(f"Unable to open {filename_indbench} for writing")
    count_indbench = 0

    ofs_removed = oechem.oemolostream()
    if not ofs_removed.open(filename_removed):
        oechem.OEThrow.Fatal(f"Unable to open {filename_removed} for writing")
    count_removed = 0

    seen_cmiles = set()
    for i, (entry_name, spec_name, rec) in enumerate(ds.iterate_records()):
        if i % 1000 == 0:
            logger.info(f"Ran {i} of {ds.record_count}")
        if spec_name != "default":
            continue
        smiles = rec.initial_molecule.extras[
            "canonical_isomeric_explicit_hydrogen_mapped_smiles"
        ]
        if smiles in seen_cmiles:
            continue
        else:
            seen_cmiles.add(smiles)
        mol = get_mol_fps(smiles)
        oechem.OESetSDData(mol, "NAME", entry_name)
        oechem.OESetSDData(mol, "CMILES", smiles)

        if rec.id in outlier_record_ids:
            oechem.OEWriteMolecule(ofs_removed, mol)
            count_removed += 1
        else:
            oechem.OEWriteMolecule(ofs_indbench, mol)
            count_indbench += 1

    logger.info(
        f"For conformers, there are {count_indbench} in v1.2 and {count_removed} removed."
    )

else:
    logger.info(f"Either {filename_indbench} and/or {filename_removed} already exists.")
