# Linear Regression — California Housing Price Prediction

## Algorithm

**Linear Regression** is a fundamental supervised learning algorithm that models the relationship between a scalar target $y$ and a vector of features $\mathbf{x}$ as a linear function:  
$\hat{y} = \mathbf{w}^\top \mathbf{x} + b$.

The model parameters $\mathbf{w}$ (coefficients) and $b$ (intercept) are learned by minimising the **Mean Squared Error** (MSE) loss:

$$L = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

Our implementation provides two solvers:

- **Normal Equation** – a closed‑form solution:  
  $\mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$
- **SGD (Stochastic Gradient Descent)** – an iterative optimiser that updates weights using mini‑batch gradients. The implementation includes:
  - **Target scaling** for stable convergence
  - **Gradient clipping** to prevent gradient explosion
  - Configurable **momentum** and **number of epochs**

## Dataset

- **Source:** California Housing dataset (`fetch_california_housing` from scikit‑learn).
- **Samples:** 20,640 instances, 8 numeric features, 1 target (median house value in $100k).
- **Preprocessing:** Features are standardised to zero mean and unit variance using `myml.utils.preprocessing.StandardScaler`. The target is scaled for SGD training (inverse transformation applied for evaluation).
- **Task:** Regression — predict the median house value of a district.

### Features (8)

| Index | Feature    |
|-------|------------|
| 0     | MedInc     |
| 1     | HouseAge   |
| 2     | AveRooms   |
| 3     | AveBedrms  |
| 4     | Population |
| 5     | AveOccup   |
| 6     | Latitude   |
| 7     | Longitude  |

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt   # from repo root
```

Make sure your `myml` package is installed in editable mode:

```bash
pip install -e .   # from repo root
```

### Run the notebook

```bash
jupyter notebook linear_regression_demo.ipynb
```

Or open in VS Code / JupyterLab and run all cells.

## Expected Outputs

- **Our Normal Equation** matches scikit‑learn’s `LinearRegression` exactly (R² ≈ 0.588, RMSE ≈ 0.745, identical coefficients).
- **Our SGD** with gradient clipping and target scaling converges close to the closed‑form solution (R² ≈ 0.585–0.588), and is on par with scikit‑learn’s `SGDRegressor` when properly configured.
- **Coefficient comparison** shows all four models agree in sign and magnitude.
- **Training loss curves** for SGD demonstrate stable convergence.
- **Prediction plots** show the fitted regression line against actual values, with R²/ RMSE / MAE printed for all methods.