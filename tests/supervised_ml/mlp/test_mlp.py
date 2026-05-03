import numpy as np
import pytest
from rice_Ml.supervised_ml.mlp.mlp import MLP
from rice_Ml.activations import ReLU, Linear, Sigmoid
from rice_Ml.loss import MeanSquaredError, BinaryCrossEntropy
from rice_Ml.optimizers import SGD

class TestMLP:
    def test_mlp_regression_xor(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])

        model = MLP(
            hidden_layers=[8],
            activation=ReLU(),
            output_activation=Sigmoid(),
            loss=BinaryCrossEntropy(),
            optimizer=SGD(learning_rate=0.5, momentum=0.9),
            n_epochs=2000,
            kernel_initializer='he_uniform',
            random_state=42
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        y_pred_labels = (y_pred > 0.5).astype(int)
        accuracy = np.mean(y_pred_labels == y)
        assert accuracy == 1.0

    def test_mlp_binary_classification(self):
        """MLP can classify simple linearly separable data."""
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)

        model = MLP(
            hidden_layers=[8],
            activation=ReLU(),
            output_activation=Sigmoid(),
            loss=BinaryCrossEntropy(),
            optimizer=SGD(learning_rate=0.5),
            n_epochs=200,
            batch_size=16,
            random_state=42
        )
        model.fit(X, y)
        y_pred_proba = model.predict(X)
        y_pred = (y_pred_proba > 0.5).astype(int)
        accuracy = np.mean(y_pred == y)
        assert accuracy > 0.9

    def test_loss_history_decreases(self):
        X = np.random.randn(50, 3)
        y = np.random.randn(50, 1)
        model = MLP(
            hidden_layers=[5],
            activation=ReLU(),
            output_activation=Linear(),
            loss=MeanSquaredError(),
            optimizer=SGD(learning_rate=0.01),
            n_epochs=50,
            random_state=42
        )
        model.fit(X, y)
        loss = model.loss_history_
        assert loss[-1] < loss[0]

    def test_predict_shape(self):
        X = np.random.randn(30, 4)
        y = np.random.randn(30, 2)
        model = MLP(
            hidden_layers=[6],
            activation=ReLU(),
            output_activation=Linear(),
            loss=MeanSquaredError(),
            optimizer=SGD(learning_rate=0.01),
            n_epochs=10,
            random_state=42
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape

    def test_linear_classification(self):
        X = np.random.randn(200, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)

        model = MLP(
            hidden_layers=[],
            activation=ReLU(),
            output_activation=Sigmoid(),
            loss=BinaryCrossEntropy(),
            optimizer=SGD(learning_rate=0.1),
            n_epochs=500,
            batch_size=None,
            kernel_initializer='xavier_uniform',
            random_state=42
        )
        model.fit(X, y)
        score = model.score(X, y)
        assert score > 0.85

    def test_score_classification(self):
        X = np.random.randn(200, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)

        model = MLP(
            hidden_layers=[8],
            activation=ReLU(),
            output_activation=Sigmoid(),
            loss=BinaryCrossEntropy(),
            optimizer=SGD(learning_rate=0.1, momentum=0.9),
            n_epochs=500,
            batch_size=32,
            kernel_initializer='he_uniform',
            random_state=42
        )
        model.fit(X, y)
        score = model.score(X, y)
        assert score > 0.85

    def test_initializer_passed_to_layers(self):
        from rice_Ml.supervised_ml.mlp.initializers import Zeros
        X = np.random.randn(10, 4)
        y = np.random.randn(10, 1)
        model = MLP(
            hidden_layers=[5],
            activation=ReLU(),
            output_activation=Linear(),
            loss=MeanSquaredError(),
            optimizer=SGD(),
            kernel_initializer=Zeros(),
            n_epochs=1
        )
        model.fit(X, y)
        W = model.layers[0].parameters()['W']
        assert np.all(W == 0.0)
