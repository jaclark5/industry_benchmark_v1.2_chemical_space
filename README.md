# Industry Benchmark v1.2 — Chemical Space Characterization

Characterizes the chemical space of the [OpenFF Industry Benchmark Season 1 v1.2](https://zenodo.org/records/15801401) dataset using molecular fingerprints, pairwise similarity distances, HDBSCAN clustering, and distance-matrix-consistent dimensionality reduction.

## Dependencies

- OpenEye Toolkits (requires a valid license)
- `scikit-learn`, `scipy`, `numpy`, `pandas`, `matplotlib`, `seaborn`
- `umap-learn` (for step 4 UMAP embeddings)
- `yammbs` (functional group analysis)
- `qcportal` (QCArchive data access)

## Reproducing these results

```bash
git clone https://github.com/jaclark5/industry_benchmark_v1.2_chemical_space.git
cd industry_benchmark_v1.2_chemical_space
micromamba create -f environment.yaml
micromamba activate ibcs
pip install umap-learn
```

Then run `./run.sh` inside each numbered directory in order.

---

## Pipeline

### `1_import_data_process_fps/`

Downloads the v1.2 industry benchmark molecules from QCArchive, deduplicates by canonical SMILES, and computes three fingerprints per molecule using the OpenEye toolkit:

| Fingerprint | Type |
|---|---|
| Circular | Extended-connectivity (Morgan-style) |
| MACCS | 166-bit structural keys |
| Lingo | Linear-path hash |

Outputs two SDF files with fingerprints embedded as hex-string data tags: one for the in-scope v1.2 molecules and one for the v1.1 entries that were removed.

### `1b_Checkmol/`

Performs a functional group census on the v1.2 molecules (via `yammbs.checkmol`), classifies molecules as drug-like by SMARTS substructure matching against a reference backbone library, and summarizes structural properties (ring types, heteroatom counts, molecular weight).

> **Note:** Ring-size detection in `plot_summary_stats` uses `OEDetermineRingSystems`, which returns atom counts for entire fused ring systems rather than individual rings. Five- and six-membered rings within fused systems (naphthalene, indole, etc.) are undercounted as a result. This is left as-is because it does not affect the functional group census or drug-likeness classification used in the associated blog post.

### `2_distances/`

Computes full pairwise distance matrices ($1 - \text{similarity}$) for all nine combinations of fingerprint type × similarity metric (Tanimoto, Dice, Cosine), and a distance vector for each removed molecule against the v1.2 set. Distance distributions are plotted as a sanity check.

### `3_cluster/`

Runs a grid search over HDBSCAN hyperparameters (`min_cluster_size` × `min_samples`) for all nine distance matrices, on both the full dense matrix and a sparse symmetric-kNN approximation. For each combination, saves cluster label files and a cluster-size histogram. Produces heatmaps of (n_clusters, noise_fraction) vs. hyperparameters and a scatterplot of "final contenders" that pass noise ≤ 0.5 and clusters ≥ 10 thresholds.

For the sparse case, a k-distance plot is computed first: for each k in a range, each molecule's distance to its k-th nearest neighbour is sorted in descending order and plotted. The elbow in those curves indicates a natural kNN distance threshold for the sparsification step.

### `4_PCA_plot_clusters/`

Visualizes the HDBSCAN clustering results by embedding each precomputed distance matrix into 2D using UMAP, MDS, and t-SNE — all operating on the same $(1 - \text{similarity})$ distance matrix used for clustering — and saving one scatter plot per method per contender, coloured by HDBSCAN cluster label. All contenders from the step-3 `summary.csv` files that pass the noise ≤ 0.5 and clusters ≥ 10 thresholds are plotted. Outputs are written to `outputs/full/` and `outputs/sparse/`.

These three methods are used instead of PCA because they accept the precomputed distance matrix directly (`metric="precomputed"` / `dissimilarity="precomputed"`), so the embedding geometry is consistent with the Tanimoto/Dice/Cosine distances that defined the clusters. PCA operates in Euclidean bit-vector space, which is a different geometry and produces embeddings where HDBSCAN cluster boundaries are not meaningful.

| Method | Characteristic |
|---|---|
| UMAP | Preserves local neighbourhood and global topology; fast; recommended |
| MDS | Minimises stress between 2D and original distances; interpretable but O(n³) |
| t-SNE | Preserves local structure; good for sub-cluster detail; does not preserve global distances |

### `5_greatest_common_structure/`

Loads a selected set of "final contender" cluster label files from step 3, groups the v1.2 molecules by cluster, renders per-cluster PDFs for clusters with ≤ 300 members, and computes the Greatest Common Fragment (GCS) for each cluster via Maximum Common Substructure search (`GetCoreFragment`, `minbonds=5`). Saves a summary CSV of GCS SMILES per cluster and a grid PDF of all reference molecules.
