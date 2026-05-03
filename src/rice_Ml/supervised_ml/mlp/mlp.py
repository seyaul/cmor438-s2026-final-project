"""Multi-layer perceptron with configurable hidden layers, activations, loss, and optimizer."""

import numpy as np
from typing import List, Optional, Union

from .base import BaseNeuralNetwork
from .layers import Dense
from ...activations import Activation
from ...loss import Loss
from ...optimizers import Optimizer
from .initializers import Initializer
from ...metrics.classification import Accuracy
from ...metrics.regression import r2_score


class MLP(BaseNeuralNetwork):
    def __init__(
        self,
        hidden_layers: List[int],
        activation: Activation,
        output_activation: Activation,
        loss: Loss,
        optimizer: Optimizer,
        n_epochs: int = 100,
        batch_size: Optional[int] = None,
        random_state: Optional[int] = None,
        kernel_initializer: Union[Initializer, str] = 'xavier_uniform',
        output_initializer: Union[Initializer, str, None] = None,
    ):
        super().__init__()
        self.hidden_units = hidden_layers
        self.activation = activation
        self.output_activation = output_activation
        self.loss_fn = loss
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.kernel_initializer = kernel_initializer
        self.output_initializer = output_initializer or kernel_initializer
        self.loss_history_ = []

    def _build(self, input_shape: tuple) -> None:
        """Build hidden layers. Output layer is deferred to fit() once output dim is known."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        for units in self.hidden_units:
            self.layers.append(Dense(units, self.activation, kernel_initializer=self.kernel_initializer))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLP':
        if self.random_state is not None:
            np.random.seed(self.random_state)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        output_units = y.shape[1]

        if not self._is_built:
            self._build(X.shape)
            self._is_built = True

        input_dim_for_output = self.layers[-1].units if self.layers else X.shape[1]
        output_layer = Dense(output_units, self.output_activation, kernel_initializer=self.output_initializer)
        output_layer._build(input_dim_for_output)
        self.layers.append(output_layer)

        n_samples = X.shape[0]
        batch_size = self.batch_size or n_samples
        self.loss_history_ = []

        for _ in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]
            epoch_loss, n_batches = 0.0, 0

            for start in range(0, n_samples, batch_size):
                X_batch = X_shuffled[start:start + batch_size]
                y_batch = y_shuffled[start:start + batch_size]

                y_pred = self.forward(X_batch)
                epoch_loss += self.loss_fn(y_batch, y_pred)
                n_batches += 1

                self.backward(self.loss_fn.gradient(y_batch, y_pred))

                flat_params, flat_grads = {}, {}
                for layer_name, layer_params in self.parameters().items():
                    for param_name, param in layer_params.items():
                        key = f"{layer_name}_{param_name}"
                        flat_params[key] = param
                        flat_grads[key] = self.gradients()[layer_name][param_name]
                self.optimizer.update(flat_params, flat_grads)

            self.loss_history_.append(epoch_loss / n_batches)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        loss_name = self.loss_fn.__class__.__name__.lower()
        is_classification = any(n in loss_name for n in ['crossentropy', 'bce', 'hinge', 'softmax'])
        if is_classification:
            if y.ndim == 1 or y.shape[1] == 1:
                y_pred_labels = (y_pred > 0.5).astype(int).flatten()
                y_true = y.astype(int).flatten()
            else:
                y_pred_labels = np.argmax(y_pred, axis=1)
                y_true = np.argmax(y, axis=1)
            return Accuracy()(y_true, y_pred_labels)
        return r2_score(y.flatten(), y_pred.flatten())
