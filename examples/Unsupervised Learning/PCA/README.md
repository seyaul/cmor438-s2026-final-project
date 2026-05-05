# PCA — Dimensionality Reduction on the Iris Dataset

## Algorithm

**Principal Component Analysis** (PCA, Pearson, 1901) is an unsupervised linear transformation that finds a set of orthogonal axes (principal components) which maximise the variance of the projected data.  
Given a centered data matrix $\mathbf{X}_c$, PCA solves the eigenvalue problem:

$$\mathbf{C} \mathbf{v} = \lambda \mathbf{v}$$

where $\mathbf{C} = \frac{1}{n-1} \mathbf{X}_c^\top \mathbf{X}_c$ is the sample covariance matrix. The eigenvectors $\mathbf{v}_i$ (principal components) are sorted by decreasing eigenvalues $\lambda_i$, which give the variance explained by each component.

Our implementation supports:

- **Dimensionality reduction** (`n_components`).
- **Whitening** – scaling the transformed components to unit variance.
- **Inverse transform** – reconstruction of original features from the reduced space.
- **Explained variance** and **explained variance ratio**.

The API follows scikit‑learn conventions (`fit`, `transform`, `fit_transform`, `inverse_transform`).

## Dataset

- **Source:** Iris dataset (`load_iris` from scikit‑learn).
- **Samples:** 150 flowers, 4 numeric features (sepal length, sepal width, petal length, petal width).
- **Classes:** 3 balanced species (setosa, versicolor, virginica).
- **Preprocessing:** Features are standardised to zero mean and unit variance using `rice_Ml.utils.preprocessing.StandardScaler`.
- **Task:** Unsupervised dimensionality reduction — project the data onto the first two principal components and examine the explained variance.

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt   # from repo root
```

Make sure your `rice_Ml` package is installed in editable mode:

```bash
pip install -e .   # from repo root
```

### Run the notebook

```bash
jupyter notebook pca_demo.ipynb
```

Or open in VS Code / JupyterLab and run all cells.

## Expected Outputs

- **Explained variance ratios:** Our PCA and scikit‑learn’s PCA report exactly the same values; the first two components capture ~95.8% of the total variance.
- **Cumulative explained variance:** A scree plot shows the steep drop after 2 components, confirming that 2D is sufficient for visualisation.
- **2‑D projection:** The scatter plot of the first two principal components clearly separates the three Iris species.
- **Sign ambiguity:** When signs of components differ between implementations, the notebook demonstrates how to align them, producing identical visualisations.
- **Reconstruction error:** Mean squared error between original scaled data and the reconstruction from 2 components is small and matches scikit‑learn’s result.
- The custom `rice_Ml.unsupervised_ml.decomposition.PCA` class is validated.