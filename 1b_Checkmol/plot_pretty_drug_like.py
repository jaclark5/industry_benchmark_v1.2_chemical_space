import json
import os

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
import cairosvg
from matplotlib.backends.backend_pdf import PdfPages
import math
import matplotlib.pyplot as plt


def _get_drug_patterns() -> dict[str, tuple[str, Chem.Mol]]:
    drug_smiles = json.load(open("reference/drug_backbones_rdkit_pretty.json", "r"))
    output = {}
    for name, smiles in drug_smiles.items():
        mol = Chem.MolFromSmarts(smiles)
        output[name] = (smiles, mol)
    return output


DRUG_PATTERNS = _get_drug_patterns()


def save_molecule_as_svg(mol, filename=None, width=400, height=400):
    """Save an RDKit molecule as a high-quality SVG file or return as image object."""

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    svg_content = drawer.GetDrawingText()

    if filename:
        with open(filename, "w") as f:
            f.write(svg_content)

    # Convert SVG to PIL Image
    png_data = cairosvg.svg2png(bytestring=svg_content.encode())
    image = Image.open(io.BytesIO(png_data))

    return image


os.makedirs("drug_patterns", exist_ok=True)
images = []
for name, (smiles, mol) in DRUG_PATTERNS.items():
    try:
        image = save_molecule_as_svg(mol, f"drug_patterns/{name}.svg")
    except Exception:
        print(f"Structure {name} failed")
    else:
        images.append(image)

# Calculate grid dimensions
num_images = len(images)
cols = math.ceil(math.sqrt(num_images))
rows = math.ceil(num_images / cols)
# Create PDF with tiled images
with PdfPages("drug_patterns/drug_patterns_tiled.pdf") as pdf:
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Flatten axes array for easier indexing
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].axis("off")
        # Get the drug name for the title
        drug_name = list(DRUG_PATTERNS.keys())[i]
        axes[i].set_title(drug_name, fontsize=12)

    # Hide empty subplots
    for i in range(len(images), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()
