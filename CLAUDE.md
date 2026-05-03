# CLAUDE.md — Rice CMOR 438 Final Project

## Project Goal
Build `rice_ML`, a Python package implementing supervised and unsupervised ML algorithms from scratch, applied to the GuitarSet audio dataset for monophonic (and eventually polyphonic) guitar signal identification.

---

## Quality Standard

**Primary reference: [rykerdolese/Data-Science-and-Machine-Learning](https://github.com/rykerdolese/Data-Science-and-Machine-Learning)**
The professor's overall gold standard. Follow this for repo structure, packaging, test layout, and CI.

**Notebook reference: [gwenfitz/fitzsimmons-cmor-438](https://github.com/gwenfitz/fitzsimmons-cmor-438)**
Has the best individual notebook quality. Follow this for how notebooks should be written.

### What makes rykerdolese the standard
- Full Python packaging: `setup.py` + `pytest.ini` + `pyproject.toml` + `requirements.txt`
- Each example has its own folder with README + bundled dataset + notebook
- Notebooks include: intuition section, full LaTeX math, EDA, decision boundary / loss curve visualizations, sklearn correctness comparison, parameter tuning, and prose commentary on every output
- CI runs on every PR

### What makes gwenfitz the notebook standard
Gwenfitz notebooks go further than rykerdolese on documentation quality:
- Numbered sections (`## 1. Load Data`, `## 2. EDA`, `## 3. Model`, etc.)
- Full LaTeX derivations for every equation, not just the final formula
- EDA section with correlation heatmap and feature importance interpretation
- Written interpretation after every output cell explaining what the result means
- Coefficient/weight interpretation tied back to domain knowledge

Use gwenfitz's notebook structure as the template whenever writing a new example notebook.

---

## Golden Standard File Trees

### ryker — rykerdolese/Data-Science-and-Machine-Learning

```
Data-Science-and-Machine-Learning/
├── .github/
│   └── workflows/
├── .gitignore
├── README.md
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── setup.py
├── src/
│   └── mlpackage/
│       ├── __init__.py
│       ├── metrics.py
│       ├── preprocess.py
│       ├── supervised_learning/
│       │   ├── Decision Trees (Classifier & Regressor)/
│       │   ├── Ensembles/
│       │   ├── KNN/
│       │   ├── Linear Regression/
│       │   ├── Logistic Regression/
│       │   ├── Neural Networks/
│       │   └── Perceptron/
│       └── unsupervised_learning/
│           ├── DBSCAN/
│           ├── K-means/
│           ├── Label Propagation - Community Detection/
│           ├── PCA/
│           └── SVD/
├── examples/
│   ├── README.md
│   ├── Supervised Learning/
│   └── Unsupervised Learning/
└── tests/
    ├── __init__.py
    ├── test_dbscan.py
    ├── test_decision_tree_classifier.py
    ├── test_decision_tree_regressor.py
    ├── test_kmeans.py
    ├── test_knn.py
    ├── test_layer_propogation.py
    ├── test_linear_regression.py
    ├── test_logistic_regression.py
    ├── test_metrics.py
    ├── test_mlp.py
    ├── test_pca.py
    ├── test_perceptron.py
    └── test_preprocess.py
```

Note: ryker uses flat tests (all `test_*.py` at `tests/` root). We prefer nested to mirror `src/`.

---

### gwen — gwenfitz/fitzsimmons-cmor-438

```
fitzsimmons-cmor-438/
├── .github/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   └── rice_ML/
│       ├── __init__.py
│       ├── supervised_learning/
│       │   ├── __init__.py
│       │   ├── classification/
│       │   │   ├── decision_trees.py
│       │   │   ├── ensemble_methods.py
│       │   │   ├── knn.py
│       │   │   ├── logistic_regression.py
│       │   │   ├── perceptron.py
│       │   │   └── README.md
│       │   └── regression/
│       │       ├── gradient_descent.py
│       │       ├── linear_regression.py
│       │       ├── multilayer_perceptron.py
│       │       ├── regression_trees.py
│       │       └── README.md
│       ├── unsupervised_learning/
│       │   ├── __init__.py
│       │   ├── clustering/
│       │   │   ├── kmeans.py
│       │   │   ├── dbscan.py
│       │   │   ├── community_detection.py
│       │   │   └── README.md
│       │   └── dimensionality_reduction/
│       │       ├── pca.py
│       │       └── README.md
│       └── processing/
│           ├── __init__.py
│           ├── preprocessing.py
│           └── postprocessing.py
├── notebooks/
│   ├── supervised_learning/
│   └── unsupervised_learning/
└── tests/
    └── unit/
```

---

## Repository Layout

```
.
├── examples/
│   ├── Supervised Learning/
│   │   └── <Algorithm>/
│   │       ├── README.md
│   │       ├── <dataset>.csv or .npz
│   │       └── <algorithm>_from_scratch.ipynb
│   └── Unsupervised Learning/
│       └── <Algorithm>/
│           ├── README.md
│           ├── <dataset>
│           └── <algorithm>_from_scratch.ipynb
├── src/
│   └── rice_ML/
│       ├── __init__.py
│       ├── metrics.py
│       ├── preprocessing/
│       ├── supervised_ml/
│       └── unsupervised_ml/
├── tests/
│   ├── __init__.py
│   ├── test_smoke.py
│   └── <module>/
│       └── test_<module>.py     ← mirrors src layout
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Adding a New Algorithm

1. **Implement** the algorithm as a class in the appropriate `src/rice_ML/supervised_ml/` or `src/rice_ML/unsupervised_ml/` subdirectory.
2. **Export** it from the relevant `__init__.py`.
3. **Write a test file** mirroring the source layout — e.g. `src/rice_ML/supervised_ml/knn/` → `tests/supervised_ml/knn/test_knn.py`. Cover fit, predict, and edge cases.
4. **Add an example** at `examples/<Supervised or Unsupervised> Learning/<Algorithm>/` with:
   - `README.md` explaining the algorithm and how to run the notebook
   - A dataset file
   - A notebook `<algorithm>_from_scratch.ipynb` showing end-to-end usage
5. **Add a smoke test** to `tests/test_smoke.py` — one import check and one end-to-end fit/predict sanity check.
6. **Run tests** with `pytest -v` before opening a PR.

---

## Testing

- Framework: `pytest`
- Run all tests: `pytest -v`
- Tests mirror the `src/` layout — `tests/<module>/test_<module>.py`
- `tests/test_smoke.py` stays at the top level for fast end-to-end sanity checks
- CI runs on every PR targeting `main` via `.github/workflows/tests.yml`

---

## Code Standards

- Algorithms must be implemented from scratch using only `numpy`, `pandas`, `scipy`, and `matplotlib` — no `sklearn` implementations in `src/`
- `sklearn` may be used in tests for correctness comparison
- Line length: 100 characters (`ruff` + `black` configured in `pyproject.toml`)
- No commented-out code; no placeholder `pass` blocks in merged code
- Docstrings: one-line summary only — no multi-paragraph blocks
