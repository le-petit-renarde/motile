"""
motile.py — Motile organism with an evolvable neural-network brain.

Each Motile is a microscopic agent inhabiting a 3D continuous grid.
Its genome IS the weights of a tiny 7→6→4 feed-forward network
(tanh activations).  No backprop, no training — weights evolve via
Gaussian mutation during asexual reproduction.
"""

import numpy as np
from typing import Optional, Tuple

from config import DEFAULT_CONFIG, SimConfig

# ---------------------------------------------------------------------------
#  Brain
# ---------------------------------------------------------------------------

class MotileBrain:
    """Feed-forward neural network whose weights are the organism's genome.

    Architecture
    ------------
    input (7)  →  hidden (6) tanh  →  output (4) tanh
    ~50 trainable parameters packed into a single flat array so that
    mutation is a simple per-element Gaussian perturbation.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration (provides layer sizes).
    weights : np.ndarray | None
        Pre-packed weight vector (e.g. inherited from a parent).
        If *None* random Xavier-like initialisation is used.
    """

    __slots__ = ("config", "n_in", "n_h", "n_out",
                 "w1", "b1", "w2", "b2", "weights")

    def __init__(self,
                 config: SimConfig = DEFAULT_CONFIG,
                 weights: Optional[np.ndarray] = None) -> None:
        self.config = config
        self.n_in = config.nn_inputs      # 7
        self.n_h = config.nn_hidden       # 6
        self.n_out = config.nn_outputs    # 4

        if weights is not None:
            self.weights = weights.astype(np.float32, copy=True)
        else:
            # Xavier-like init for tanh
            w1_std = np.sqrt(2.0 / (self.n_in + self.n_h))
            w2_std = np.sqrt(2.0 / (self.n_h + self.n_out))
            rng = np.random.default_rng()
            self.w1 = rng.standard_normal(
                (self.n_in, self.n_h), dtype=np.float32) * w1_std
            self.b1 = np.zeros(self.n_h, dtype=np.float32)
            self.w2 = rng.standard_normal(
                (self.n_h, self.n_out), dtype=np.float32) * w2_std
            self.b2 = np.zeros(self.n_out, dtype=np.float32)
            self.weights = self._pack()

    # ------------------------------------------------------------------
    #  Packing / unpacking
    # ------------------------------------------------------------------

    def _pack(self) -> np.ndarray:
        """Flatten all parameters into a single 1-D array (the genome)."""
        return np.concatenate([
            self.w1.ravel(),
            self.b1,
            self.w2.ravel(),
            self.b2,
        ]).astype(np.float32)

    def _unpack(self) -> None:
        """Rebuild w1, b1, w2, b2 from the packed weight vector."""
        w1_end = self.n_in * self.n_h
        b1_end = w1_end + self.n_h
        w2_end = b1_end + self.n_h * self.n_out
        self.w1 = self.weights[:w1_end].reshape(self.n_in, self.n_h)
        self.b1 = self.weights[w1_end:b1_end]
        self.w2 = self.weights[b1_end:w2_end].reshape(self.n_h, self.n_out)
        self.b2 = self.weights[w2_end:]

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the network on *inputs*.

        Parameters
        ----------
        inputs : np.ndarray
            Shape ``(n_in,)`` for a single sample or ``(batch, n_in)``
            for a batch.

        Returns
        -------
        np.ndarray
            Shape ``(n_out,)`` for single, ``(batch, n_out)`` for batch.
        """
        self._unpack()
        x = np.atleast_2d(np.asarray(inputs, dtype=np.float32))
        h = np.tanh(x @ self.w1 + self.b1)
        out = np.tanh(h @ self.w2 + self.b2)
        # Squeeze back to 1-D if a single sample was supplied.
        if out.shape[0] == 1:
            return out[0].astype(np.float32)
        return out.astype(np.float32)

    # ------------------------------------------------------------------
    #  Mutation helpers
    # ------------------------------------------------------------------

    def scramble(self, fraction: float = 0.25) -> None:
        """Perturb a random subset of weights — simulates phage disruption.

        Parameters
        ----------
        fraction : float
            Proportion of weights to perturb (0.0 – 1.0).
        """
        n = len(self.weights)
        mask = np.random.random(n) < fraction
        noise = np.random.randn(np.sum(mask)).astype(np.float32) * np.float32(0.5)
        self.weights[mask] += noise
        np.clip(self.weights, np.float32(-3.0), np.float32(3.0), out=self.weights)


# ---------------------------------------------------------------------------
#  Organism
# ---------------------------------------------------------------------------

# Colour mapping used by get_status() for terminal / GUI visualisation.
_STATUS_COLORS: dict = {
    "HEALTHY":   "\033[92m",   # bright green
    "INFECTED":  "\033[91m",   # bright red
    "RESISTANT": "\033[94m",   # bright blue
}
_RESET = "\033[0m"


class Motile:
    """A single organism driven by an evolvable neural-network brain.

    Life cycle
    ----------
    1. **Sense**  – sample local chemical gradients and internal state.
    2. **Decide** – run the brain → ``[dX, dY, dZ, action]``.
    3. **Act**    – simulation moves the motile, updates energy, etc.
    4. **Divide** – if energy permits, produce a mutated child.

    Attributes
    ----------
    config : SimConfig
    position : np.ndarray  (3,) float32
    velocity : np.ndarray  (3,) float32
    energy : float
    age : int
    generation : int
    resistance : float    [0, 1]
    infectivity : float   [0, 1]
    status : str          HEALTHY | INFECTED | RESISTANT
    brain : MotileBrain
    id : int
    infection_ticks : int
    """

    __slots__ = (
        "config", "position", "velocity", "energy", "age", "generation",
        "resistance", "infectivity", "status", "brain", "id",
        "infection_ticks",
    )

    _next_id: int = 0

    def __init__(self,
                 config: SimConfig = DEFAULT_CONFIG,
                 position: Optional[np.ndarray] = None,
                 brain: Optional[MotileBrain] = None,
                 resistance: float = 0.1,
                 infectivity: float = 0.2,
                 energy: float = 50.0,
                 generation: int = 0) -> None:
        """Create a new motile.

        Parameters
        ----------
        config : SimConfig
        position : np.ndarray | None
            (3,) starting position. Random uniform inside grid if *None*.
        brain : MotileBrain | None
            Pre-built brain. Random initialisation when *None*.
        resistance : float
            Innate resistance to phage infection [0, 1].
        infectivity : float
            How strongly this motile sheds phage when infected [0, 1].
        energy : float
            Starting energy pool.
        generation : int
            Number of ancestral divisions.
        """
        self.config = config
        self.id = Motile._next_id
        Motile._next_id += 1

        # Position in continuous coordinates [0, grid_size)
        s = float(config.grid_size)
        if position is None:
            rng = np.random.default_rng()
            self.position = rng.uniform(0.0, s - 1.0, size=3).astype(np.float32)
        else:
            self.position = np.asarray(position, dtype=np.float32)

        self.velocity = np.zeros(3, dtype=np.float32)
        self.energy = float(energy)
        self.age = 0
        self.generation = int(generation)

        self.resistance = float(np.clip(resistance, 0.0, 1.0))
        self.infectivity = float(np.clip(infectivity, 0.0, 1.0))

        self.status = "HEALTHY"
        self.infection_ticks = 0

        self.brain = brain if brain is not None else MotileBrain(config)

    # ------------------------------------------------------------------
    #  Spatial helpers
    # ------------------------------------------------------------------

    def grid_pos(self) -> np.ndarray:
        """Integer grid cell this motile currently occupies.

        Returns
        -------
        np.ndarray
            Shape ``(3,)``, dtype int, values in ``[0, grid_size-1]``.
        """
        return np.maximum(0, np.minimum(self.config.grid_size - 1, np.rint(self.position).astype(np.int32)))

    # ------------------------------------------------------------------
    #  Perception → action pipeline
    # ------------------------------------------------------------------

    def sense(self, engine) -> np.ndarray:
        """Build the 7-element sensory input vector from the environment.

        Parameters
        ----------
        engine : SimulationEngine
            Provides ``sample_at``, ``gradient_at``, and field constants.

        Returns
        -------
        np.ndarray
            Shape ``(7,)`` float32, roughly normalised to ``[-1, 1]``.
        """
        cfg = self.config
        gp = self.grid_pos().reshape(1, 3)

        # Chemical concentrations at current cell.
        _nutrient, toxin, phage = engine.sample_at(gp)

        # Nutrient gradient (direction of increasing food).
        grad = engine.gradient_at(gp, engine.NUTRIENT)
        nx, ny, nz = grad[0]

        # Neighbour density is injected by the simulation loop later;
        # default to zero here so the brain always sees a valid value.
        neighbor_density = np.float32(0.0)

        # Pack and clip inputs to keep them in a tanh-friendly range.
        inputs = np.array([
            np.clip(nx * 5.0,   -1.0, 1.0),   # nutrient grad X
            np.clip(ny * 5.0,   -1.0, 1.0),   # nutrient grad Y
            np.clip(nz * 5.0,   -1.0, 1.0),   # nutrient grad Z
            np.clip(toxin[0] * 2.0,  -1.0, 1.0),  # toxin level
            np.clip(phage[0] * 2.0,  -1.0, 1.0),  # phage level
            np.clip(self.energy / 100.0 - 0.5, -1.0, 1.0),  # energy state
            np.clip(neighbor_density * 2.0, -1.0, 1.0),
        ], dtype=np.float32)

        return inputs

    def decide(self, inputs: np.ndarray) -> np.ndarray:
        """Feed sensory input through the brain.

        Parameters
        ----------
        inputs : np.ndarray
            Shape ``(7,)`` float32 sensory vector from ``sense()``.

        Returns
        -------
        np.ndarray
            Shape ``(4,)`` action vector: ``[dX, dY, dZ, action]``.
        """
        return self.brain.forward(inputs)

    # ------------------------------------------------------------------
    #  Metabolism & status
    # ------------------------------------------------------------------

    def metabolize(self) -> bool:
        """Consume energy for one simulation tick.

        Costs include basal metabolism, movement, infection drain,
        and a gentle aging penalty for very old organisms.

        Returns
        -------
        bool
            *True* if the motile still has positive energy, *False*
            otherwise (caller should mark it dead).
        """
        cfg = self.config
        cost = cfg.base_metabolism

        # Movement cost scales with speed.
        speed = float(np.linalg.norm(self.velocity))
        cost += speed * cfg.move_energy_cost

        # Infected motiles burn energy faster.
        if self.status == "INFECTED":
            cost += cfg.infection_energy_drain
            self.infection_ticks += 1

        # Small linear age penalty beyond 2000 ticks (discourages immortals).
        if self.age > 2000:
            cost += (self.age - 2000) * 0.001

        self.energy -= cost
        self.age += 1

        return self.energy > 0.0

    def get_status(self) -> Tuple[str, str]:
        """Human-readable status with an ANSI colour code for display.

        Returns
        -------
        tuple[str, str]
            ``(status_label, ansi_coloured_string)``.
        """
        color = _STATUS_COLORS.get(self.status, _RESET)
        colored = f"{color}{self.status}{_RESET}"
        return self.status, colored

    # ------------------------------------------------------------------
    #  Emission
    # ------------------------------------------------------------------

    def emit_toxin(self) -> float:
        """Base metabolic waste emitted into the grid per tick.

        Returns
        -------
        float
            Toxin units deposited.
        """
        return self.config.base_metabolism * 0.5

    def emit_phage(self) -> float:
        """Phage particles shed per tick (only when infected).

        Returns
        -------
        float
            Phage units deposited.
        """
        if self.status == "INFECTED":
            return self.config.phage_emit_rate * self.infectivity
        return 0.0

    # ------------------------------------------------------------------
    #  Reproduction
    # ------------------------------------------------------------------

    def can_divide(self) -> bool:
        """Check whether this motile is eligible to reproduce.

        Requirements
        ------------
        * Energy >= ``min_energy_for_division``
        * Not currently infected

        Returns
        -------
        bool
        """
        return (
            self.energy >= self.config.min_energy_for_division
            and self.status != "INFECTED"
        )

    def create_child(self) -> "Motile":
        """Produce a genetically-mutated child; parent pays the energy cost.

        Mutation strategy
        -----------------
        * 30 % of weights receive small Gaussian perturbations.
        * 1 % of weights receive a 3× larger "leap" mutation.
        * Resistance / infectivity traits are Gaussian-perturbed and
          clipped to [0, 1].  Resistant parents confer a small bonus.

        Returns
        -------
        Motile
            A new organism placed near the parent.
        """
        cfg = self.config
        self.energy -= cfg.division_cost

        # ---- brain mutation ----
        child_weights = self.brain.weights.copy()

        # Standard mutations.
        mut_mask = np.random.random(len(child_weights)) < 0.3
        child_weights[mut_mask] += (
            np.random.randn(np.sum(mut_mask)).astype(np.float32)
            * np.float32(cfg.mutation_rate)
        )

        # Rare large "leap" mutations.
        leap_mask = np.random.random(len(child_weights)) < 0.01
        child_weights[leap_mask] += (
            np.random.randn(np.sum(leap_mask)).astype(np.float32)
            * np.float32(cfg.mutation_rate)
            * np.float32(3.0)
        )

        np.clip(child_weights, np.float32(-3.0), np.float32(3.0),
                out=child_weights)

        child_brain = MotileBrain(cfg, weights=child_weights)

        # ---- trait mutation ----
        child_resist = self.resistance + np.random.normal(
            0.0, cfg.resistance_mutation)
        if self.status == "RESISTANT":
            child_resist += 0.05
        child_resist = np.clip(child_resist, 0.0, 1.0)

        child_infect = self.infectivity + np.random.normal(
            0.0, cfg.resistance_mutation)
        child_infect = np.clip(child_infect, 0.0, 1.0)

        # ---- position ----
        offset = np.random.randn(3).astype(np.float32) * np.float32(1.5)
        child_pos = self.position + offset
        np.clip(child_pos, 0.0, float(cfg.grid_size - 1), out=child_pos)

        return Motile(
            config=cfg,
            position=child_pos,
            brain=child_brain,
            resistance=float(child_resist),
            infectivity=float(child_infect),
            energy=cfg.child_energy,
            generation=self.generation + 1,
        )

    # ------------------------------------------------------------------
    #  Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Lightweight snapshot for logging / replay / visualisation.

        Returns
        -------
        dict
        """
        return {
            "id": self.id,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "energy": round(self.energy, 2),
            "age": self.age,
            "generation": self.generation,
            "resistance": round(self.resistance, 3),
            "infectivity": round(self.infectivity, 3),
            "status": self.status,
            "infection_ticks": self.infection_ticks,
        }