"""
Unit tests for CNN and its layer primitives.
"""
import numpy as np
import pytest
from rice_Ml.supervised_ml.cnn.layers import Conv1D, MaxPool1D, Flatten, Dense
from rice_Ml.supervised_ml.cnn.cnn import CNN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_clf_data(rng):
    """Binary classification dataset with 18 features (matches GuitarSet default)."""
    X = rng.standard_normal((120, 18))
    y = rng.integers(0, 2, size=120)
    return X, y


@pytest.fixture
def multi_clf_data(rng):
    """5-class dataset."""
    X = rng.standard_normal((200, 18))
    y = rng.integers(0, 5, size=200)
    return X, y


# ---------------------------------------------------------------------------
# Conv1D
# ---------------------------------------------------------------------------

class TestConv1D:
    def test_output_shape(self, rng):
        layer = Conv1D(in_channels=1, out_channels=16, kernel_size=3)
        X = rng.standard_normal((8, 1, 18))
        out = layer.forward(X)
        assert out.shape == (8, 16, 16)

    def test_backward_shape(self, rng):
        layer = Conv1D(in_channels=1, out_channels=16, kernel_size=3)
        X = rng.standard_normal((8, 1, 18))
        out = layer.forward(X)
        dout = rng.standard_normal(out.shape)
        dX = layer.backward(dout)
        assert dX.shape == X.shape

    def test_grad_shapes(self, rng):
        layer = Conv1D(in_channels=1, out_channels=16, kernel_size=3)
        X = rng.standard_normal((4, 1, 18))
        out = layer.forward(X)
        layer.backward(rng.standard_normal(out.shape))
        assert layer.grads["W"].shape == layer.W.shape
        assert layer.grads["b"].shape == layer.b.shape

    def test_params_keys(self):
        layer = Conv1D(1, 8, 3)
        assert set(layer.params.keys()) == {"W", "b"}


# ---------------------------------------------------------------------------
# MaxPool1D
# ---------------------------------------------------------------------------

class TestMaxPool1D:
    def test_output_shape(self, rng):
        layer = MaxPool1D(pool_size=2)
        X = rng.standard_normal((8, 16, 16))
        out = layer.forward(X)
        assert out.shape == (8, 16, 8)

    def test_selects_max(self):
        layer = MaxPool1D(pool_size=2)
        X = np.array([[[1.0, 3.0, 2.0, 4.0]]])  # (1, 1, 4)
        out = layer.forward(X)
        np.testing.assert_array_equal(out, [[[3.0, 4.0]]])

    def test_backward_shape(self, rng):
        layer = MaxPool1D(pool_size=2)
        X = rng.standard_normal((4, 16, 16))
        out = layer.forward(X)
        dX = layer.backward(rng.standard_normal(out.shape))
        assert dX.shape == X.shape

    def test_gradient_flows_to_max(self):
        layer = MaxPool1D(pool_size=2)
        X = np.array([[[1.0, 3.0]]])  # max is index 1
        layer.forward(X)
        dX = layer.backward(np.ones((1, 1, 1)))
        assert dX[0, 0, 0] == 0.0   # non-max gets 0
        assert dX[0, 0, 1] == 1.0   # max gets full gradient


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

class TestFlatten:
    def test_forward_shape(self, rng):
        layer = Flatten()
        X = rng.standard_normal((8, 32, 6))
        out = layer.forward(X)
        assert out.shape == (8, 192)

    def test_backward_restores_shape(self, rng):
        layer = Flatten()
        X = rng.standard_normal((8, 32, 6))
        out = layer.forward(X)
        dX = layer.backward(rng.standard_normal(out.shape))
        assert dX.shape == X.shape


# ---------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------

class TestDense:
    def test_output_shape(self, rng):
        layer = Dense(192, 64)
        X = rng.standard_normal((8, 192))
        assert layer.forward(X).shape == (8, 64)

    def test_backward_input_shape(self, rng):
        layer = Dense(192, 64)
        X = rng.standard_normal((8, 192))
        out = layer.forward(X)
        dX = layer.backward(rng.standard_normal(out.shape))
        assert dX.shape == X.shape

    def test_grad_shapes(self, rng):
        layer = Dense(10, 5)
        X = rng.standard_normal((4, 10))
        out = layer.forward(X)
        layer.backward(rng.standard_normal(out.shape))
        assert layer.grads["W"].shape == (10, 5)
        assert layer.grads["b"].shape == (5,)


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------

class TestCNN:
    def test_fit_returns_self(self, small_clf_data):
        X, y = small_clf_data
        model = CNN(epochs=2, random_state=0)
        assert model.fit(X, y) is model

    def test_predict_shape(self, small_clf_data):
        X, y = small_clf_data
        model = CNN(epochs=2, random_state=0).fit(X, y)
        assert model.predict(X).shape == (len(X),)

    def test_predict_proba_shape_and_sums(self, small_clf_data):
        X, y = small_clf_data
        model = CNN(epochs=2, random_state=0).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_labels_in_classes(self, small_clf_data):
        X, y = small_clf_data
        model = CNN(epochs=2, random_state=0).fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset(set(model.classes_))

    def test_score_range(self, small_clf_data):
        X, y = small_clf_data
        model = CNN(epochs=2, random_state=0).fit(X, y)
        s = model.score(X, y)
        assert 0.0 <= s <= 1.0

    def test_loss_history_length(self, small_clf_data):
        X, y = small_clf_data
        model = CNN(epochs=5, random_state=0).fit(X, y)
        assert len(model.loss_history_) == 5

    def test_loss_decreases(self, rng):
        """Loss should trend down on a learnable dataset over enough epochs."""
        X = rng.standard_normal((300, 18))
        # Linearly separable: class = sign of first feature
        y = (X[:, 0] > 0).astype(int)
        model = CNN(epochs=20, learning_rate=1e-2, batch_size=32, random_state=0)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_multiclass(self, multi_clf_data):
        X, y = multi_clf_data
        model = CNN(epochs=2, random_state=0).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 5)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            CNN().predict_proba(np.ones((5, 18)))

    def test_bad_input_dims_raises(self, small_clf_data):
        X, y = small_clf_data
        with pytest.raises(ValueError):
            CNN(epochs=1).fit(X.ravel(), y)

    def test_too_few_features_raises(self):
        X = np.ones((10, 4))
        y = np.zeros(10, dtype=int)
        with pytest.raises(ValueError, match="too small"):
            CNN(epochs=1).fit(X, y)

    def test_repr(self):
        model = CNN(learning_rate=0.01, epochs=10, batch_size=32)
        assert "CNN" in repr(model)
        assert "0.01" in repr(model)
