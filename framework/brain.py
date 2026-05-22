import numpy as np
from typing import List, Optional
from .base import Brain

class FeedForwardBrain(Brain):
    """
    A generic multi-layer feed-forward neural network brain.
    Weights are stored in a single flat array to easily allow for evolutionary mutation.
    """
    def __init__(self, layer_sizes: List[int], weights: Optional[np.ndarray] = None):
        if len(layer_sizes) < 2:
            raise ValueError("Brain must have at least an input and an output layer")

        self.layer_sizes = layer_sizes

        # Calculate total number of weights and biases
        self.total_weights = 0
        self._shapes = []
        for i in range(len(layer_sizes) - 1):
            w_size = layer_sizes[i] * layer_sizes[i + 1]
            b_size = layer_sizes[i + 1]
            self.total_weights += w_size + b_size
            self._shapes.append((layer_sizes[i], layer_sizes[i + 1]))

        if weights is not None:
            if weights.size != self.total_weights:
                raise ValueError(f"Expected {self.total_weights} weights, got {weights.size}")
            self.weights = weights.astype(np.float32, copy=True)
        else:
            self.weights = np.zeros(self.total_weights, dtype=np.float32)
            self._initialize_weights()

    def _initialize_weights(self):
        """Xavier-like initialization for tanh activation."""
        rng = np.random.default_rng()
        offset = 0
        for i, (n_in, n_out) in enumerate(self._shapes):
            w_std = np.sqrt(2.0 / (n_in + n_out))
            w = rng.standard_normal((n_in, n_out), dtype=np.float32) * w_std
            b = np.zeros(n_out, dtype=np.float32)

            w_size = n_in * n_out
            b_size = n_out

            self.weights[offset:offset+w_size] = w.ravel()
            offset += w_size
            self.weights[offset:offset+b_size] = b
            offset += b_size

    def _unpack(self) -> List[tuple[np.ndarray, np.ndarray]]:
        """Rebuild weight and bias matrices from the flat array."""
        unpacked = []
        offset = 0
        for n_in, n_out in self._shapes:
            w_size = n_in * n_out
            b_size = n_out

            w = self.weights[offset:offset+w_size].reshape((n_in, n_out))
            offset += w_size

            b = self.weights[offset:offset+b_size]
            offset += b_size

            unpacked.append((w, b))
        return unpacked

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the feed-forward network using tanh activations."""
        x = np.atleast_2d(np.asarray(inputs, dtype=np.float32))
        layers = self._unpack()

        for i, (w, b) in enumerate(layers):
            x = np.tanh(x @ w + b)

        if x.shape[0] == 1:
            return x[0].astype(np.float32)
        return x.astype(np.float32)

    def mutate(
        self,
        mutation_rate: float,
        *,
        mutation_prob: float = 0.3,
        leap_prob: float = 0.01,
        leap_multiplier: float = 3.0,
    ) -> None:
        """
        Mutate weights.
        - `mutation_prob`: fraction of weights to mutate standardly.
        - `leap_prob`: fraction to mutate with a larger jump.
        """
        n = len(self.weights)

        # Standard mutations
        mut_mask = np.random.random(n) < mutation_prob
        self.weights[mut_mask] += (
            np.random.randn(np.sum(mut_mask)).astype(np.float32) * np.float32(mutation_rate)
        )

        # Leap mutations
        leap_mask = np.random.random(n) < leap_prob
        self.weights[leap_mask] += (
            np.random.randn(np.sum(leap_mask)).astype(np.float32)
            * np.float32(mutation_rate)
            * np.float32(leap_multiplier)
        )

        np.clip(self.weights, -3.0, 3.0, out=self.weights)

    def copy(self) -> 'FeedForwardBrain':
        return FeedForwardBrain(self.layer_sizes, self.weights.copy())
