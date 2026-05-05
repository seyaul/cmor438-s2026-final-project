# DBSCAN — Density‑Based Clustering on Synthetic Data

## Algorithm

**DBSCAN** (Ester et al., 1996) is a density‑based clustering algorithm. It groups together points that are closely packed while marking points in low‑density regions as noise. The algorithm relies on two parameters:

- **`eps` (ε):** Maximum distance between two samples for one to be considered in the neighbourhood of the other.
- **`min_samples`:** Minimum number of points required to form a dense region (core point).

DBSCAN does **not** require specifying the number of clusters in advance, can find arbitrarily shaped clusters, and is robust to outliers (noise points). It works by iteratively expanding clusters from core points through density‑reachable neighbourhoods.

## Dataset

- **Source:** Synthetic data generated with scikit‑learn’s `make_blobs` and `make_moons`.
- **Blobs dataset:** 300 samples, 3 well‑separated spherical clusters, 2 features.
- **Moons dataset:** 300 samples, 2 interleaving half‑circles (non‑convex), 2 features, 5% noise.
- **Preprocessing:** Both datasets are standardised to zero mean and unit variance using `rice_Ml.utils.preprocessing.StandardScaler`.
- **Task:** Unsupervised clustering — recover the underlying group structure and identify noise points.

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt   # from repo root (numpy, matplotlib, seaborn, scikit‑learn, jupyter)
```

Ensure your `rice_Ml` package is installed in editable mode:

```bash
pip install -e .   # from repo root
```

### Run the notebook

```bash
jupyter notebook dbscan_demo.ipynb
```

Or open in VS Code / JupyterLab and run all cells.

## Expected Outputs

- **Cluster recovery (Blobs):** Both custom and scikit‑learn DBSCAN find exactly 3 clusters with minimal noise (< 5%). The ARI between the two implementations exceeds 0.99, confirming identical behaviour.
- **Cluster recovery (Moons):** DBSCAN successfully identifies the two non‑convex moon shapes, separating them accurately. Noise points are isolated along the boundary.
- **Visualisation:** Side‑by‑side scatter plots show nearly identical cluster assignments for both implementations. Noise points are displayed in grey.
- **Quantitative agreement:** Adjusted Rand Index (ARI) > 0.99 for both datasets, demonstrating that the custom implementation matches scikit‑learn’s output.