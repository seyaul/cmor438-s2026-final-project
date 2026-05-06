# KNN — Diabetes Classification & Decision-Boundary Exploration

## Algorithm

**K-Nearest Neighbours (KNN)** is an instance-based, non-parametric algorithm.  It stores the entire training set and predicts by polling the *k* closest points.

### Distance

The default metric is Euclidean distance between feature vectors **x**ᵢ and **x**ⱼ ∈ ℝᵖ:

```
d(xᵢ, xⱼ) = sqrt( Σₘ (xᵢₘ − xⱼₘ)² )
```

The implementation also supports **taxicab** (Manhattan) distance and any user-supplied callable.

### Classification rule

Let 𝒩ₖ(**x**) be the indices of the *k* nearest training points.  The predicted class is the majority vote:

```
ŷ = argmax_c  Σᵢ∈𝒩ₖ(x)  𝟙(yᵢ = c)
```

### Regression rule

For continuous targets, prediction is the (possibly distance-weighted) mean:

```
ŷ = (1/k)  Σᵢ∈𝒩ₖ(x)  yᵢ          (uniform weights)
ŷ = Σᵢ (yᵢ/dᵢ) / Σᵢ (1/dᵢ)        (distance weights)
```

### Key trade-offs

| Aspect | Detail |
|---|---|
| Training time | O(1) — stores data only |
| Prediction time | O(n · p) per query — expensive at large n |
| Memory | Grows with dataset size |
| Scaling sensitivity | High — features must be standardised |
| Hyperparameters | k (neighbourhood size), distance metric |

## Dataset

- **Source:** [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (NIDDK)
- **File:** `diabetes.csv`
- **Samples:** 768 patients (500 negative, 268 positive)
- **Task:** Binary classification — predict diabetes onset (Outcome = 0 or 1)

### Features (8)

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (2-hour oral tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (μU/mL) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Genetic risk score |
| Age | Age in years |

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt   # from repo root
```

### Run the notebook

```bash
jupyter notebook knn_from_scratch.ipynb
```

`diabetes.csv` is bundled in this folder — no download required.

## Expected Outputs

- **EDA scatter plots:** Glucose vs. BMI and correlation heatmap showing class overlap.
- **2-feature decision boundary:** Mesh plot over Glucose × BMI space; smaller *k* produces jagged, over-fitted regions; larger *k* smooths them.
- **Effect of scaling:** Unscaled KNN underperforms because Glucose (~0–200) dominates BMI (~0–70) in distance calculations; StandardScaler equalises feature contributions.
- **k-tuning curve:** Accuracy vs. *k* (1–30); optimal *k* is typically 9–15 on this dataset.
- **Full-feature test accuracy:** ~75–79% with k ≈ 11 and StandardScaler.
- **sklearn comparison:** `sklearn.neighbors.KNeighborsClassifier` with identical k and metric should match our accuracy within ~1%.

## Notes

- Zero values for Glucose, BloodPressure, SkinThickness, Insulin, and BMI are physiologically impossible and represent missing data; the notebook replaces them with column medians before fitting.
- **Feature scaling is essential** for KNN — always apply `StandardScaler` before fitting.
- The `KNNClassifier` also exposes `predict_proba` (fractional vote counts) which can be used with ROC-AUC evaluation.
- `KNNRegressor` supports `weights="distance"` for inverse-distance-weighted predictions.
