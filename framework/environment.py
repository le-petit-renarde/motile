import numpy as np
from typing import Tuple, List, Optional
from .base import Environment, Agent

class GridEnvironment(Environment):
    """
    A generic 3D grid environment with multiple diffusion channels.
    Provides vectorized methods to compute diffusion, decay, and query gradients/samples.
    """
    def __init__(self, size: int, num_channels: int, diffusion_rates: List[float], decay_rates: List[float]):
        self.size = size
        self.num_channels = num_channels

        if len(diffusion_rates) != num_channels or len(decay_rates) != num_channels:
            raise ValueError("diffusion_rates and decay_rates must match num_channels")

        self.diffusion_rates = np.array(diffusion_rates, dtype=np.float32)
        # Convert decay rates to per-tick linear decay multipliers (1.0 - decay)
        self.decay_factors = 1.0 - np.array(decay_rates, dtype=np.float32)

        # Grid: [channel, x, y, z]
        self.grid = np.zeros((self.num_channels, self.size, self.size, self.size), dtype=np.float32)

        # We don't need a full kernel for convolution, just manual slices for speed like in engine.py
        # Center = 1.0, 6 face neighbors = 0.6 -> kernel sum = 1.0 + 6*0.6 = 4.6
        self._kernel_sum = 4.6

    def reset(self) -> None:
        self.grid.fill(0.0)

    def sample_at(self, positions: np.ndarray) -> np.ndarray:
        """
        Sample all channels at the given positions.
        positions: (N, 3) float array.
        Returns: (N, num_channels) float array.
        """
        positions = np.clip(positions, 0, self.size - 1).astype(np.intp)
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

        # Result: shape (num_channels, N)
        samples = self.grid[:, x, y, z]
        # Transpose to (N, num_channels)
        return samples.T

    def gradient_at(self, positions: np.ndarray, channel_idx: int, eps: float = 1.0) -> np.ndarray:
        """
        Compute central-difference gradient of a specific channel.
        positions: (N, 3) array.
        Returns: (N, 3) array of [dx, dy, dz].
        """
        positions = np.clip(positions, 1, self.size - 2).astype(np.intp)
        field = self.grid[channel_idx]
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

        dx = (field[x + 1, y, z] - field[x - 1, y, z]) * (0.5 / eps)
        dy = (field[x, y + 1, z] - field[x, y - 1, z]) * (0.5 / eps)
        dz = (field[x, y, z + 1] - field[x, y, z - 1]) * (0.5 / eps)

        return np.column_stack([dx, dy, dz]).astype(np.float32)

    def _diffuse(self, field: np.ndarray, rate: float) -> np.ndarray:
        """One tick of face-neighbor diffusion on a 3D scalar field."""
        if rate <= 0:
            return field

        w = 0.6  # face-neighbor weight
        conv = field.copy()

        # Accumulate the six face neighbors using slicing
        conv[1:, :, :] += w * field[:-1, :, :]
        conv[:-1, :, :] += w * field[1:, :, :]
        conv[:, 1:, :] += w * field[:, :-1, :]
        conv[:, :-1, :] += w * field[:, 1:, :]
        conv[:, :, 1:] += w * field[:, :, :-1]
        conv[:, :, :-1] += w * field[:, :, 1:]

        conv *= (1.0 / self._kernel_sum)

        # Blend: new = old + rate * (convolved - old)
        return field + rate * (conv - field)

    def step_chemistry(self, emissions: Optional[List[Tuple[int, np.ndarray, np.ndarray]]] = None) -> None:
        """
        Advance the chemical fields.
        emissions: list of tuples (channel_idx, positions, amounts)
        """
        # 1. Diffusion
        for ch in range(self.num_channels):
            rate = self.diffusion_rates[ch]
            if rate > 0:
                self.grid[ch] = self._diffuse(self.grid[ch], rate)

        # 2. Decay
        for ch in range(self.num_channels):
            self.grid[ch] *= self.decay_factors[ch]

        # 3. Add emissions
        if emissions:
            for ch, positions, amounts in emissions:
                if len(positions) > 0 and len(amounts) > 0:
                    positions = np.clip(positions, 0, self.size - 1).astype(np.intp)
                    np.add.at(
                        self.grid[ch],
                        (positions[:, 0], positions[:, 1], positions[:, 2]),
                        amounts
                    )

        # 4. Clamp to [0, 1]
        np.clip(self.grid, 0.0, 1.0, out=self.grid)

    def step(self, agents: List[Agent]) -> None:
        """
        Generic environment step hook.
        Child classes should override this to apply custom logic (like nutrient influx)
        and then call step_chemistry() with the appropriate emissions.
        """
        raise NotImplementedError
