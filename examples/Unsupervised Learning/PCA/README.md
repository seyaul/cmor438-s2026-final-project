# PCA — Pizza Nutritional Composition Analysis

## Algorithm  

**Principal Component Analysis** (PCA, Pearson, 1901) is an unsupervised linear transformation that identifies the directions (principal components) of maximum variance in the data. Given a centered data matrix $\mathbf{X}_c$, the principal components are the eigenvectors of the sample covariance matrix  

$$\mathbf{C} = \frac{1}{n-1} \mathbf{X}_c^\top \mathbf{X}_c$$

sorted by decreasing eigenvalues, which indicate the variance explained by each component.  

Our implementation supports:  

- **Dimensionality reduction** via `n_components`.  
- **Whitening** (optional).  
- **Inverse transform** to reconstruct original features.  
- Computation of **explained variance** and **explained variance ratio**.  

The API follows scikit‑learn conventions (`fit`, `transform`, `fit_transform`, `inverse_transform`).

## Dataset  

- **Source:** `pizza.csv` – nutritional content of pizzas from different brands.  
- **Samples:** 300 pizzas, 10 brands (A–J).  
- **Features (7 numeric):** moisture, protein, fat, ash, sodium, carbohydrates, calories.  
- **Categorical field:** `brand` (used for visualisation only, not for clustering).  
- **Preprocessing:**  
  - Features are standardised to zero mean and unit variance with `myml.utils.preprocessing.StandardScaler`.  
  - Identifier column `id` is dropped.  
- **Task:** Unsupervised dimensionality reduction – discover which nutritional characteristics dominate the variance and how different brands separate in the reduced space.

## How to Run  

### Prerequisites  

```bash
pip install -r requirements.txt   # from repo root (numpy, pandas, matplotlib, seaborn, scikit‑learn, jupyter)
```

Make sure your `myml` package is installed in editable mode:

```bash
pip install -e .   # from repo root
```

Place `pizza.csv` in the same folder as the notebook (or adjust the path).

### Run the notebook  

```bash
jupyter notebook pca_pizza_demo.ipynb
```

Or open in VS Code / JupyterLab and run all cells.

## Expected Outputs  

- **Exploratory Data Analysis:**  
  - Count plot showing the distribution of samples across brands.  
  - Pairplot and correlation heatmap revealing strong links between moisture, protein, fat, and carbs.  
- **Explained Variance:**  
  - Our PCA and scikit‑learn’s PCA report identical explained variance ratios.  
  - A scree plot illustrates that the first two components capture a substantial fraction of the total variance.  
- **2‑D Projection:**  
  - The first two principal components separate several brands (e.g., brand I forms a distinct cluster).  
  - After aligning the arbitrary signs (since eigenvectors can flip), the custom and scikit‑learn scatter plots are visually identical.  
- **Component Loadings:**  
  - Heatmap showing how strongly each original feature contributes to PC1 and PC2 (e.g., moisture vs. fat contrast).  
- **Reconstruction Error:**  
  - Mean squared error between the original scaled data and the reconstruction from 2 components is small and matches scikit‑learn.  

This validates the `myml.unsupervised_ml.decomposition.PCA` class on a real, multi‑feature dataset with natural groupings.