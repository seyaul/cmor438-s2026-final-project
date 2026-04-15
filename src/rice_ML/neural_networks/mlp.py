"""
MLP Implementation
"""

import numpy as np
from typing import List, Optional, Union

from .base import BaseNeuralNetwork
from .layers import Dense
from ..activations import Activation
from ..loss import Loss
from ..optimizers import Optimizer
from .initializers import Initializer
from ..metrics.classification import Accuracy
from ..metrics.regression import r2_score


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
        output_initializer: Union[Initializer, str, None] = None
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
        """
        Build only the hidden layers.
        The output layer will be built in fit() once the output dimension is known.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        input_dim = input_shape[1]
        for units in self.hidden_units:
            self.layers.append(Dense(
                units,
                self.activation,
                kernel_initializer=self.kernel_initializer
            ))
        # Output layer is intentionally NOT created here.

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLP':
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Determine output shape from y
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        output_units = y.shape[1]

        # Build hidden layers if not already built
        if not self._is_built:
            self._build(X.shape)
            self._is_built = True

        # Build the output layer now that we know the number of units
        input_dim_for_output = self.layers[-1].units if self.layers else X.shape[1]
        output_layer = Dense(
            output_units,
            self.output_activation,
            kernel_initializer=self.output_initializer
        )
        output_layer._build(input_dim_for_output)
        self.layers.append(output_layer)

        n_samples = X.shape[0]
        batch_size = self.batch_size or n_samples

        self.loss_history_ = []

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward
                y_pred = self.forward(X_batch)
                loss = self.loss_fn(y_batch, y_pred)
                epoch_loss += loss
                n_batches += 1

                # Backward
                dloss = self.loss_fn.gradient(y_batch, y_pred)
                self.backward(dloss)

                # Gather parameters and gradients
                flat_params = {}
                flat_grads = {}
                for layer_name, layer_params in self.parameters().items():
                    for param_name, param in layer_params.items():
                        key = f"{layer_name}_{param_name}"
                        flat_params[key] = param
                        flat_grads[key] = self.gradients()[layer_name][param_name]

                # Diagnostic check (only once)
                if epoch == 0 and start == 0:
                    first_key = list(flat_params.keys())[0]
                    w_before = flat_params[first_key].copy()

                if epoch % 100 == 0 and start == 0:
                    total_grad_norm = sum(np.linalg.norm(g) for g in flat_grads.values())
                    avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
                    print(f"Epoch {epoch}: loss={avg_loss:.6f}, grad_norm={total_grad_norm:.6f}")

                # Update parameters
                self.optimizer.update(flat_params, flat_grads)

                if epoch == 0 and start == 0:
                    w_after = flat_params[first_key]
                    if np.allclose(w_before, w_after):
                        print("WARNING: Weights did not change after optimizer update!")
                    else:
                        print("Weights updated successfully.")

            self.loss_history_.append(epoch_loss / n_batches)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)

        # Infer task type from loss function name (more robust than checking values)
        loss_name = self.loss_fn.__class__.__name__.lower()
        is_classification = any(name in loss_name for name in
                                ['crossentropy', 'bce', 'hinge', 'softmax'])

        if is_classification:
            if y.ndim == 1 or y.shape[1] == 1:
                y_pred_labels = (y_pred > 0.5).astype(int)
                y_true = y.astype(int).flatten()
            else:
                y_pred_labels = np.argmax(y_pred, axis=1)
                y_true = np.argmax(y, axis=1)
            acc = Accuracy()
            return acc(y_true, y_pred_labels)
        else:
            # Regression
            return r2_score(y.flatten(), y_pred.flatten())