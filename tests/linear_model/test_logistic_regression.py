import numpy as np
import pytest
from rice_Ml.supervised_ml.linear_model import LogisticRegression
from rice_Ml.metrics.classification import accuracy

class TestLogisticRegression:
    """Unit tests for the LogisticRegression class."""

    def test_fit_predict_binary_linear_separable(self):
        """Model should perfectly separate linearly separable binary data."""
        X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]])
        y = np.array([0, 0, 0, 1, 1, 1])

        model = LogisticRegression(learning_rate=0.5, n_epochs=500, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)

        assert accuracy(y, y_pred) == 1.0
        assert model.loss_history_[-1] < 0.1

    def test_predict_proba_shape_and_range(self):
        """predict_proba should return values in [0, 1] with correct shape."""
        X = np.random.randn(20, 3)
        y = np.random.randint(0, 2, 20)

        model = LogisticRegression(n_epochs=100, random_state=42)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (20,)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_intercept_handling(self):
        """fit_intercept=False should force intercept to zero."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 0, 1, 1])

        model = LogisticRegression(fit_intercept=False, n_epochs=300, random_state=42)
        model.fit(X, y)

        assert model.intercept_ == 0.0

        # When x=0, the linear combination is 0 (since intercept is 0).
        # Sigmoid(0) = 0.5 exactly.
        proba_zero = model.predict_proba(np.array([[0.0]]))
        assert proba_zero[0] == pytest.approx(0.5, abs=0.01)

    def test_convergence_loss_decreases(self):
        """Loss should strictly decrease over epochs."""
        X = np.random.randn(50, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticRegression(learning_rate=0.1, n_epochs=100, random_state=42)
        model.fit(X, y)

        loss = model.loss_history_
        # Overall decreasing trend (allow small fluctuations due to SGD)
        assert loss[-1] < loss[0]
        # At least some improvement
        assert loss[-1] < loss[0] * 0.9

    def test_predict_threshold(self):
        """Changing threshold should affect predictions."""
        X = np.random.randn(30, 2)
        y = np.random.randint(0, 2, 30)

        model = LogisticRegression(n_epochs=100, random_state=42)
        model.fit(X, y)

        y_pred_default = model.predict(X)
        y_pred_low = model.predict(X, threshold=0.1)
        y_pred_high = model.predict(X, threshold=0.9)

        # Low threshold → more positive predictions
        assert np.sum(y_pred_low) >= np.sum(y_pred_default)
        # High threshold → fewer positive predictions
        assert np.sum(y_pred_high) <= np.sum(y_pred_default)

    def test_gradient_clipping_stability(self):
        """Model with clipping should train without overflow on unscaled data."""
        X = np.random.randn(100, 5) * 10  # large scale
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticRegression(
            learning_rate=0.1, n_epochs=50, clipnorm=1.0, random_state=42
        )
        model.fit(X, y)

        assert np.isfinite(model.loss_history_[-1])
        assert np.all(np.isfinite(model.coef_))

    def test_momentum_accelerates(self):
        """Momentum should help converge faster."""
        X = np.random.randn(80, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model_no_mom = LogisticRegression(
            learning_rate=0.1, n_epochs=100, momentum=0.0, random_state=42
        )
        model_no_mom.fit(X, y)

        model_mom = LogisticRegression(
            learning_rate=0.1, n_epochs=100, momentum=0.9, random_state=42
        )
        model_mom.fit(X, y)

        # Momentum should achieve lower or equal final loss
        assert model_mom.loss_history_[-1] <= model_no_mom.loss_history_[-1] + 0.05

    def test_random_state_reproducibility(self):
        """Same random_state should produce identical results."""
        X = np.random.randn(40, 3)
        y = np.random.randint(0, 2, 40)

        model1 = LogisticRegression(n_epochs=50, random_state=42)
        model1.fit(X, y)

        model2 = LogisticRegression(n_epochs=50, random_state=42)
        model2.fit(X, y)

        np.testing.assert_array_almost_equal(model1.coef_, model2.coef_)
        assert model1.intercept_ == model2.intercept_

    def test_invalid_hyperparameters_raise(self):
        """Invalid hyperparameters should raise ValueError."""
        with pytest.raises(ValueError):
            LogisticRegression(learning_rate=-0.1)

        with pytest.raises(ValueError):
            LogisticRegression(n_epochs=0)

        with pytest.raises(ValueError):
            LogisticRegression(momentum=1.5)

        with pytest.raises(ValueError):
            LogisticRegression(clipnorm=-1.0)

    def test_score_method_accuracy(self):
        """score() should return accuracy."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 1])

        model = LogisticRegression(n_epochs=200, random_state=42)
        model.fit(X, y)

        acc = model.score(X, y)
        y_pred = model.predict(X)
        expected_acc = accuracy(y, y_pred)

        assert acc == pytest.approx(expected_acc)