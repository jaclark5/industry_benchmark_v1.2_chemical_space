# Purpose:
Characterize the chemical space for the [Industry Benchmark v1.2](https://zenodo.org/records/15801401) dataset.

## In scope:
- Evaluate fingerprints of the molecules in the dataset using the open-eye toolkit
- Cluster using distances based off of similarity metrics from the open-eye toolkit
- Visualize clustering using dimensionality reduction techniques

## Repeating these results
1. Create the relevant conda environment
```bash
git clone https://github.com/jaclark5/industry_benchmark_v1.2_chemical_space.git
```
2. Install needed conda env with:
```bash
cd industry_benchmark_v1.2_chemical_space; micromamba create -f environment.yaml
```
3. Run ``main.py`` files in each step denoted by `#_*`
