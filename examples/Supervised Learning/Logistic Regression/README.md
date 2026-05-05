# Logistic Regression — Breast Cancer Wisconsin Classification

## Algorithm

**Logistic Regression** is a linear binary classifier. It models the probability that a sample belongs to the positive class as:

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b)$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function.  
The parameters $\mathbf{w}$ and $b$ are learned by minimising the **Binary Cross‑Entropy** (BCE) loss:

$$L = -\frac{1}{n}\sum_{i=1}^n \big[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \big]$$

Our implementation uses **Stochastic Gradient Descent** (SGD) with the following features:

- **Feature standardisation** for stable convergence.
- **Gradient clipping** to prevent exploding gradients.
- Configurable **momentum** and **number of epochs**.
- **Sigmoid activation** for probability outputs.
- **Threshold‑based classification** (default 0.5).

The model inherits from `BaseLinearModel` and reuses the same modular components (loss, optimiser, metrics) as the linear regression module.

## Dataset

- **Source:** Breast Cancer Wisconsin Diagnostic Dataset (`load_breast_cancer` from scikit‑learn).
- **Samples:** 569 instances, 30 numeric features, 1 binary target (0 = malignant, 1 = benign).
- **Class balance:** ~37% malignant, ~63% benign.
- **Preprocessing:** Features are standardised to zero mean and unit variance using `myml.utils.preprocessing.StandardScaler`.
- **Task:** Binary classification — distinguish malignant from benign breast tumours.

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt   # from repo root
```

Ensure your `myml` package is installed in editable mode:

```bash
pip install -e .   # from repo root
```

### Run the notebook

```bash
jupyter notebook logistic_regression_demo.ipynb
```

Or open in VS Code / JupyterLab and run all cells.

## Expected Outputs

- **Our logistic regression** achieves high accuracy (~97–98%), closely matching scikit‑learn’s `LogisticRegression` (with no penalty).
- **Precision, Recall, F1** scores are reported and are comparable across implementations.
- **Training loss curve** shows smooth convergence of the binary cross‑entropy loss.
- **Coefficient comparison** of the top five features shows similar magnitudes and identical signs.
- **Confusion matrix** heatmap visualises the classification performance (true vs. predicted labels).
- Both implementations agree almost perfectly, validating the custom `LogisticRegression` class.