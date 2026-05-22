"""
engine.py — 3D artificial‑life simulation engine for the Motile project.

Manages a cubic grid of three diffusing chemical fields (nutrient, toxin,
phage) and provides vectorised, NumPy‑only routines for advancing physics,
sampling concentrations, computing gradients, and snapshotting state.

Requirements: numpy (no GPU, no ML frameworks, no scipy).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from config import DEFAULT_CONFIG, SimConfig


class SimulationEngine:
    """Core simulation engine managing the 3D chemical grid.

    The grid is a ``(3, N, N, N)`` float32 array where channel 0 is
    nutrient, channel 1 is toxin, and channel 2 is phage.  All per‑tick
    operations (diffusion, emission, decay, boundary influx) are fully
    vectorised with no Python‑level loops over grid cells.

    Parameters
    ----------
    config : SimConfig
        Simulation hyper‑parameters.  Defaults to ``DEFAULT_CONFIG``.
    """

    # Channel indices for readability.
    NUTRIENT: int = 0
    TOXIN: int = 1
    PHAGE: int = 2

    # Per‑channel diffusion rates (set in __init__).
    _DIFF_RATES: Tuple[float, float, float]

    # Per‑channel decay factors (set in __init__).
    _DECAY_FACTORS: Tuple[float, float, float]

    def __init__(self, config: SimConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.size = config.grid_size

        # ---- 3‑channel grid: (NUTRIENT, TOXIN, PHAGE) × N × N × N -----------
        self.grid = np.zeros(
            (3, self.size, self.size, self.size), dtype=np.float32
        )

        # Per‑channel look‑up tables for tick_chemistry.
        self._DIFF_RATES = (
            config.chem_diffusion,
            config.chem_diffusion,
            config.phage_diffusion,
        )
        self._DECAY_FACTORS = (
            1.0 - 0.0005,         # nutrient decays very slowly
            1.0 - config.chem_decay,
            1.0 - config.phage_decay,
        )

        # ---- Pre‑compute diffusion kernel & normaliser -----------------------
        self._build_diffusion_kernel()

    # ------------------------------------------------------------------
    # Kernel helpers
    # ------------------------------------------------------------------

    def _build_diffusion_kernel(self) -> None:
        """Build the 3×3×3 face‑neighbour diffusion stencil.

        Weights: centre = 1.0, each of the 6 face neighbours = 0.6.
        Diagonal / edge neighbours are set to zero for performance.
        The kernel is normalised so that convolving a uniform field
        leaves it unchanged.
        """
        kernel = np.zeros((3, 3, 3), dtype=np.float32)
        centre = 1.0
        face = 0.6

        kernel[1, 1, 1] = centre  # centre
        kernel[1, 1, 0] = face    # -z
        kernel[1, 1, 2] = face    # +z
        kernel[1, 0, 1] = face    # -y
        kernel[1, 2, 1] = face    # +y
        kernel[0, 1, 1] = face    # -x
        kernel[2, 1, 1] = face    # +x

        self._kernel = kernel
        self._kernel_sum: float = float(kernel.sum())  # 4.6

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the grid and seed the bottom‑half with initial nutrient."""
        self.grid.fill(0.0)
        # Nutrient in lower half of the y‑axis (y < size//2).
        self.grid[self.NUTRIENT, : self.size // 2, :, :] = 0.3

    def tick_chemistry(
        self,
        motile_positions: Optional[np.ndarray] = None,
        motile_toxin_emit: Optional[np.ndarray] = None,
        motile_phage_emit: Optional[np.ndarray] = None,
    ) -> None:
        """Advance the chemical fields by one simulation tick.

        Order of operations:
        1. Diffuse each channel independently.
        2. Apply nutrient influx at the top boundary (planes y=0,1,2).
        3. Apply exponential decay per channel.
        4. Accumulate motile emissions into toxin / phage fields.
        5. Clip all values to [0, 1].

        Parameters
        ----------
        motile_positions : (M, 3) int ndarray or None
            Grid coordinates of each motile (x, y, z).
        motile_toxin_emit : (M,) float ndarray or None
            Toxin quantity emitted by each motile this tick.
        motile_phage_emit : (M,) float ndarray or None
            Phage quantity emitted by each motile this tick.
        """
        # 1. Diffusion ---------------------------------------------------------
        for ch in (self.NUTRIENT, self.TOXIN, self.PHAGE):
            rate = self._DIFF_RATES[ch]
            if rate > 0:
                self.grid[ch] = self._diffuse(self.grid[ch], rate)

        # 2. Nutrient influx from top (small y = "top" of the world) -----------
        self.grid[self.NUTRIENT, 0:3, :, :] += self.config.chem_nutrient_influx

        # 3. Decay -------------------------------------------------------------
        for ch in (self.NUTRIENT, self.TOXIN, self.PHAGE):
            self.grid[ch] *= self._DECAY_FACTORS[ch]

        # 4. Motile emissions --------------------------------------------------
        if motile_positions is not None and len(motile_positions) > 0:
            x, y, z = motile_positions[:, 0], motile_positions[:, 1], motile_positions[:, 2]
            if motile_toxin_emit is not None and len(motile_toxin_emit) > 0:
                np.add.at(self.grid[self.TOXIN], (x, y, z), motile_toxin_emit)
            if motile_phage_emit is not None and len(motile_phage_emit) > 0:
                np.add.at(self.grid[self.PHAGE], (x, y, z), motile_phage_emit)

        # 5. Clamp -------------------------------------------------------------
        np.clip(self.grid, 0.0, 1.0, out=self.grid)

    def sample_at(
        self, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample chemical concentrations at each supplied position.

        Parameters
        ----------
        positions : (M, 3) int ndarray
            Grid indices (x, y, z).  Values are clamped to valid bounds.

        Returns
        -------
        nutrient : (M,) float32 ndarray
        toxin : (M,) float32 ndarray
        phage : (M,) float32 ndarray
        """
        positions = np.clip(positions, 0, self.size - 1).astype(np.intp)
        nutrient = self.grid[self.NUTRIENT, positions[:, 0], positions[:, 1], positions[:, 2]]
        toxin = self.grid[self.TOXIN, positions[:, 0], positions[:, 1], positions[:, 2]]
        phage = self.grid[self.PHAGE, positions[:, 0], positions[:, 1], positions[:, 2]]
        return nutrient, toxin, phage

    def gradient_at(
        self, positions: np.ndarray, chem_idx: int, eps: float = 1.0
    ) -> np.ndarray:
        """Compute central‑difference gradient of a chemical field.

        Parameters
        ----------
        positions : (M, 3) float ndarray
            Query positions.  Clamped to [1, size‑2] to keep stencil in bounds.
        chem_idx : int
            Channel index (NUTRIENT=0, TOXIN=1, PHAGE=2).
        eps : float
            Step size (default 1.0 = one grid cell).

        Returns
        -------
        grad : (M, 3) float32 ndarray
            Gradient vectors [dx, dy, dz] at each position.
        """
        positions = np.clip(positions, 1, self.size - 2).astype(np.intp)
        field = self.grid[chem_idx]
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

        dx = (field[x + 1, y, z] - field[x - 1, y, z]) * (0.5 / eps)
        dy = (field[x, y + 1, z] - field[x, y - 1, z]) * (0.5 / eps)
        dz = (field[x, y, z + 1] - field[x, y, z - 1]) * (0.5 / eps)

        return np.column_stack([dx, dy, dz]).astype(np.float32)

    def get_snapshot(self) -> dict:
        """Return a lightweight, JSON‑serialisable summary of the current grid.

        Returns
        -------
        dict
            Keys: ``nutrient_mean``, ``toxin_mean``, ``phage_mean``,
            ``nutrient_max``, ``toxin_max``, ``phage_max``.
        """
        return {
            "nutrient_mean": float(self.grid[self.NUTRIENT].mean()),
            "toxin_mean": float(self.grid[self.TOXIN].mean()),
            "phage_mean": float(self.grid[self.PHAGE].mean()),
            "nutrient_max": float(self.grid[self.NUTRIENT].max()),
            "toxin_max": float(self.grid[self.TOXIN].max()),
            "phage_max": float(self.grid[self.PHAGE].max()),
        }

    def save_grid(self, tick: int) -> str:
        """Persist the full grid to a compressed ``.npz`` file.

        Parameters
        ----------
        tick : int
            Current simulation tick (used in the filename).

        Returns
        -------
        str
            Absolute path to the saved file.
        """
        out_dir = self.config.output_dir
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"grid_{tick:06d}.npz")
        np.savez_compressed(path, grid=self.grid, tick=tick)
        return os.path.abspath(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _diffuse(self, field: np.ndarray, rate: float) -> np.ndarray:
        """One tick of face‑neighbour diffusion on a 3D scalar field.

        Uses a weighted sum of the centre cell and its 6 orthogonal
        neighbours, normalised by ``_kernel_sum``.  Boundaries are handled
        implicitly by weighting missing neighbours as zero (constant
        padding).

        Implemented with pure NumPy slicing so that **scipy is not
        required**.

        Parameters
        ----------
        field : (N, N, N) float32 ndarray
        rate : float
            Diffusion rate ∈ [0, 1].

        Returns
        -------
        (N, N, N) float32 ndarray
        """
        S = self.size
        w = 0.6  # face‑neighbour weight (must match _build_diffusion_kernel)

        # Start with the centre contribution (weight = 1.0).
        conv = field.copy()

        # Accumulate the six face neighbours.
        # +x  (neighbour at x-1 contributes to cell at x)
        conv[1:, :, :] += w * field[:-1, :, :]
        # -x  (neighbour at x+1 contributes to cell at x)
        conv[:-1, :, :] += w * field[1:, :, :]
        # +y
        conv[:, 1:, :] += w * field[:, :-1, :]
        # -y
        conv[:, :-1, :] += w * field[:, 1:, :]
        # +z
        conv[:, :, 1:] += w * field[:, :, :-1]
        # -z
        conv[:, :, :-1] += w * field[:, :, 1:]

        conv *= (1.0 / self._kernel_sum)

        # Blend: new = old + rate * (convolved - old)
        return field + rate * (conv - field)


# ----------------------------------------------------------------------
# Quick smoke‑test (run with ``python engine.py``)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Motile Engine Smoke Test ===")

    engine = SimulationEngine()
    engine.reset()

    print(f"Grid shape : {engine.grid.shape}")
    print(f"Initial nutrient mean : {engine.grid[engine.NUTRIENT].mean():.4f}")

    # Run 20 ticks of chemistry with no motiles.
    for t in range(20):
        engine.tick_chemistry()

    snap = engine.get_snapshot()
    print("\nAfter 20 ticks (no motiles):")
    for k, v in snap.items():
        print(f"  {k:20s} {v:.6f}")

    # Test sampling.
    positions = np.array([[10, 10, 10], [40, 20, 30]], dtype=np.intp)
    n, tx, ph = engine.sample_at(positions)
    print(f"\nSample at [10,10,10] → nutrient={n[0]:.4f} toxin={tx[0]:.4f} phage={ph[0]:.4f}")

    # Test gradient.
    grad = engine.gradient_at(positions, engine.NUTRIENT)
    print(f"Nutrient gradient at [10,10,10] : {grad[0]}")

    # Test with motile emissions.
    motile_pos = np.array([[24, 24, 24], [25, 25, 25]], dtype=np.intp)
    toxin_emit = np.array([0.5, 0.2], dtype=np.float32)
    phage_emit = np.array([0.1, 0.3], dtype=np.float32)
    engine.tick_chemistry(motile_pos, toxin_emit, phage_emit)
    snap2 = engine.get_snapshot()
    print(f"\nAfter motile emissions → toxin_mean={snap2['toxin_mean']:.6f}")

    # Save grid.
    path = engine.save_grid(42)
    print(f"\nGrid saved → {path}")

    # Reload and verify.
    data = np.load(path)
    assert data["tick"] == 42
    print("Reload OK.")

    print("\n=== All smoke tests passed ===")