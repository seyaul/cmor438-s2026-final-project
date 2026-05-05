# Decision Tree — Binary Diabetes Classification

## Algorithm

A **Decision Tree** partitions the feature space by recursively choosing the split $(j, t)$ that maximises **information gain**:

```
IG(S, j, t) = Impurity(S) − [ |S_L|/|S| · Impurity(S_L) + |S_R|/|S| · Impurity(S_R) ]
```

where $S_L = \{x \in S : x_j \le t\}$ and $S_R$ is the complement. Two impurity measures are supported:

**Gini impurity:**
```
Gini(S) = 1 − Σ_k p_k²
```

**Entropy:**
```
H(S) = −Σ_k p_k log₂(p_k)
```

Splitting continues until a node is pure, contains fewer than `min_samples_split` samples, or `max_depth` is reached. At each leaf, the majority class label is assigned.

The implementation supports `max_features` subsampling (`None`, `int`, `"sqrt"`, `"log2"`) to enable use as a base learner inside Random Forest.

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
jupyter notebook decision_tree_from_scratch.ipynb
```

`diabetes.csv` is bundled in this folder — no download required.

## Expected Outputs

- **Decision boundary plots:** 2-D slices (Glucose vs. BMI) showing how depth controls partition granularity.
- **Test accuracy:** ~72–77% depending on `max_depth`; `max_depth=5` with `criterion="gini"` is a strong baseline.
- **Depth vs. accuracy curve:** Accuracy rises sharply up to depth 4–5 then plateaus or dips (overfitting).
- **sklearn comparison:** `sklearn.tree.DecisionTreeClassifier` with the same hyperparameters should match our accuracy within ~1%.
- **Feature importances:** Glucose and BMI consistently rank as the two most discriminative features.

## Notes

- Zero values for Glucose, BloodPressure, SkinThickness, Insulin, and BMI are physiologically impossible and represent missing data; the notebook replaces them with column medians before fitting.
- `max_features` is set to `"sqrt"` in the Random Forest notebook to keep splits diverse across trees.
