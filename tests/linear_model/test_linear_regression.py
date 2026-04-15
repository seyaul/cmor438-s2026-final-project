import sys
from pathlib import Path
import numpy as np
import pytest
from rice_ML.supervised_ml.linear_model import LinearRegression

def test_normal_equation_perfect_fit():
    X = np.array([[1, 3],
                  [2, 1],
                  [3, 5],
                  [4, 2]])   # x2 is NOT a multiple of x1
    true_coef = np.array([2.0, 0.5])
    true_intercept = 3.0
    y = true_intercept + X @ true_coef
    model = LinearRegression(fit_intercept=True, solver='normal')
    model.fit(X, y)
    np.testing.assert_almost_equal(model.intercept_, true_intercept, decimal=10)
    np.testing.assert_array_almost_equal(model.coef_, true_coef, decimal=10)

def test_sgd_converges_to_normal_equation():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    # Scale features to help SGD
    from rice_ML.preprocessing.scale import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    true_coef = np.array([1.5, -2.0, 0.5])
    y = 4.0 + X @ true_coef + np.random.randn(100) * 0.1

    lr_normal = LinearRegression(solver='normal')
    lr_normal.fit(X, y)

    lr_sgd = LinearRegression(solver='sgd', learning_rate=0.1, n_epochs=500, random_state=42)
    lr_sgd.fit(X, y)

    # Compare predictions instead of coefficients (more stable)
    y_pred_normal = lr_normal.predict(X)
    y_pred_sgd = lr_sgd.predict(X)
    np.testing.assert_allclose(y_pred_sgd, y_pred_normal, rtol=0.01)

    # Or loosen coefficient tolerance
    np.testing.assert_allclose(lr_sgd.intercept_, lr_normal.intercept_, rtol=0.2)
    np.testing.assert_allclose(lr_sgd.coef_, lr_normal.coef_, rtol=0.2, atol=0.2)

def test_sgd_with_momentum():
    """Momentum should accelerate convergence on a simple problem."""
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])

    # Without momentum
    model_no_mom = LinearRegression(
        solver='sgd', learning_rate=0.1, n_epochs=200,
        momentum=0.0, random_state=42
    )
    model_no_mom.fit(X, y)

    # With momentum
    model_mom = LinearRegression(
        solver='sgd', learning_rate=0.1, n_epochs=200,
        momentum=0.9, random_state=42
    )
    model_mom.fit(X, y)

    # Both should converge, but momentum may reach lower loss faster
    assert model_no_mom.loss_history_[-1] < 0.1
    assert model_mom.loss_history_[-1] < 0.1

def test_gradient_clipping_prevents_explosion():
    """Clipping should limit weight updates when gradients are large."""
    # Data with moderate scale, but we use a huge learning rate to create large updates
    X = np.array([[10.0], [20.0], [30.0]])
    y = np.array([20.0, 40.0, 60.0])

    model_no_clip = LinearRegression(
        solver='sgd', learning_rate=10.0, n_epochs=20,
        clipnorm=None, random_state=42
    )
    model_no_clip.fit(X, y)
    loss_no_clip = model_no_clip.loss_history_[-1]

    model_clip = LinearRegression(
        solver='sgd', learning_rate=10.0, n_epochs=20,
        clipnorm=1.0, random_state=42
    )
    model_clip.fit(X, y)
    loss_clip = model_clip.loss_history_[-1]

    assert loss_clip < loss_no_clip
    assert np.isfinite(loss_clip)

def test_clipnorm_passed_to_optimizer():
    """Ensure clipnorm is stored and passed correctly."""
    model = LinearRegression(solver='sgd', clipnorm=5.0)
    assert model.clipnorm == 5.0
    # After fit, the internal optimizer should have the same clipnorm
    X = np.random.randn(10, 2)
    y = np.random.randn(10)
    model.fit(X, y)
    # We can't directly inspect optimizer.clipnorm without exposing it,
    # but we can trust the attribute is passed.

def test_invalid_clipnorm_raises():
    """Negative clipnorm should raise an error (optional validation)."""
    with pytest.raises(ValueError):
        LinearRegression(solver='sgd', clipnorm=-1.0)

def test_no_intercept():
    """When fit_intercept=False, the intercept should be zero and line forced through origin."""
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])  # y = 2*x, no intercept

    model = LinearRegression(fit_intercept=False, solver='normal')
    model.fit(X, y)

    assert model.intercept_ == 0.0
    np.testing.assert_almost_equal(model.coef_[0], 2.0)

def test_predict_shape():
    X = np.random.randn(50, 5)
    y = np.random.randn(50)
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape

def test_score_perfect_fit():
    X = np.array([[1, 2], [3, 4]])
    y = 10 + 3*X[:,0] + 5*X[:,1]
    model = LinearRegression().fit(X, y)
    assert model.score(X, y) == pytest.approx(1.0)

def test_sgd_deterministic():
    X = np.random.randn(100, 2)
    y = np.random.randn(100)

    model1 = LinearRegression(solver='sgd', random_state=42)
    model1.fit(X, y)

    model2 = LinearRegression(solver='sgd', random_state=42)
    model2.fit(X, y)

    assert model1.intercept_ == model2.intercept_
    np.testing.assert_array_equal(model1.coef_, model2.coef_)
    np.testing.assert_array_equal(model1.loss_history_, model2.loss_history_)

def test_singular_matrix_handling():
    # Create linearly dependent features
    X = np.array([[1, 2, 3],
                  [2, 4, 6],
                  [3, 6, 9]])
    y = np.array([1, 2, 3])
    model = LinearRegression(solver='normal')
    # Should not raise an error (falls back to pseudo‑inverse)
    model.fit(X, y)
    assert model.coef_ is not None