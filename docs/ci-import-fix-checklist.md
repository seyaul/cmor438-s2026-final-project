# Checklist: Fix CI import errors (`rice_ML` vs `rice_ML`)

Context: GitHub Actions job **74058368004** fails during pytest collection with:
`ModuleNotFoundError: No module named 'rice_ML'`.

Goal: Standardize imports and coverage target to the actual package name **`rice_ML`**.

## 1) Update GitHub Actions workflow
- [ ] Edit `.github/workflows/tests.yml`
- [ ] Change the coverage module from `rice_ML` to `rice_ML`:
  - [ ] Replace:
    - `pytest --cov=rice_ML --cov-report=term-missing`
  - [ ] With:
    - `pytest --cov=rice_ML --cov-report=term-missing`

## 2) Update test imports to match package name
- [ ] Search tests for incorrect casing:
  - [ ] `rg "rice_ML" -n tests`
- [ ] Replace all `rice_ML` imports with `rice_ML`
  - [ ] Example: in `tests/test_cnn.py`
    - [ ] Replace `from rice_ML.supervised_ml.cnn.layers import ...`
    - [ ] With `from rice_ML.supervised_ml.cnn.layers import ...`
- [ ] Do the same for KNN tests:
  - [ ] `tests/knn/test_classifier.py`
  - [ ] `tests/knn/test_recommender.py`
  - [ ] `tests/knn/test_regressor.py`

## 3) Update any internal library imports (src/) if needed
- [ ] Search the source tree for incorrect casing:
  - [ ] `rg "rice_ML" -n src`
- [ ] Replace any `rice_ML` references with `rice_ML`

## 4) Verify locally
- [ ] Create a clean env and install:
  - [ ] `python -m pip install -U pip`
  - [ ] `pip install -e ".[test]"`
- [ ] Run tests:
  - [ ] `pytest -q`
- [ ] Run coverage (optional):
  - [ ] `pytest --cov=rice_ML --cov-report=term-missing`

## 5) Verify in GitHub Actions
- [ ] Push changes / update PR
- [ ] Confirm the **Tests** workflow passes on Python 3.10 / 3.11 / 3.12

## Notes
- Linux runners are case-sensitive, so `rice_ML` and `rice_ML` are different modules.
- `pyproject.toml` declares the project name as `rice_ML` and packages `src/rice_ML`.
