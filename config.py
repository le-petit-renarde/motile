"""
config.py — Simulation configuration for Motile 3D artificial life.

A single dataclass holding all tunable parameters for the chemical grid,
phage dynamics, motile agents, neural networks, evolution, and I/O.
"""

from dataclasses import dataclass


@dataclass
class SimConfig:
    """All parameters governing a Motile simulation run.

    Attributes
    ----------
    grid_size : int
        Side length of the cubic N×N×N grid.
    cell_volume : float
        Physical scale of one grid cell (arbitrary units).
    chem_diffusion : float
        Per‑tick diffusion rate for nutrient and toxin fields.
    chem_decay : float
        Per‑tick exponential decay factor for toxin.
    chem_nutrient_influx : float
        Amount of nutrient added to the top three y‑planes each tick.
    chem_toxin_threshold : float
        Toxin concentration above which a motile takes damage.
    phage_diffusion : float
        Per‑tick diffusion rate for the phage field.
    phage_decay : float
        Per‑tick exponential decay factor for phage.
    phage_infection_threshold : float
        Phage density required before an infection attempt can occur.
    phage_emit_rate : float
        How much phage an infected motile sheds per tick.
    initial_population : int
        Number of motiles to spawn at simulation start.
    max_population : int
        Hard population cap; excess motiles are culled.
    min_energy_for_division : float
        Energy a motile must have before it can divide.
    division_cost : float
        Energy subtracted from the parent on division.
    child_energy : float
        Starting energy of a newly budded motile.
    base_metabolism : float
        Energy burned per tick simply to stay alive.
    move_energy_cost : float
        Additional energy consumed per unit of movement distance.
    max_speed : float
        Maximum displacement (grid cells) per tick in any axis.
    nn_inputs : int
        Number of input neurons (fixed architecture).
    nn_hidden : int
        Number of hidden neurons (single hidden layer).
    nn_outputs : int
        Number of output neurons (dx, dy, dz, action).
    mutation_rate : float
        Standard deviation of Gaussian noise added to child NN weights.
    resistance_mutation : float
        Mutation magnitude for resistance / infectivity traits.
    selection_pressure : float
        Fraction of motiles culled randomly each tick when overpopulated.
    base_infection_prob : float
        Per‑tick probability of infection when phage exceeds threshold.
    infection_weight_scramble : float
        Fraction of NN weights re‑randomised on infection.
    resistance_decay : float
        How much innate resistance decays per tick (imperfect immunity).
    infection_energy_drain : float
        Extra energy consumed per tick while infected.
    recovery_threshold : float
        Base per‑tick recovery probability (scaled by resistance).
    save_every : int
        Ticks between full‑state snapshots written to disk.
    output_dir : str
        Directory where snapshots and logs are stored.
    """

    # ---- Grid ----------------------------------------------------------------
    grid_size: int = 48
    cell_volume: float = 1.0

    # ---- Chemicals -----------------------------------------------------------
    chem_diffusion: float = 0.12
    chem_decay: float = 0.002
    chem_nutrient_influx: float = 0.5
    chem_toxin_threshold: float = 0.6

    # ---- Phage ---------------------------------------------------------------
    phage_diffusion: float = 0.12
    phage_decay: float = 0.0002
    phage_infection_threshold: float = 0.10
    phage_emit_rate: float = 1.0

    # ---- Motile population ---------------------------------------------------
    initial_population: int = 150
    max_population: int = 350
    min_energy_for_division: float = 80.0
    division_cost: float = 50.0
    child_energy: float = 30.0
    base_metabolism: float = 0.3
    move_energy_cost: float = 0.02
    max_speed: float = 0.6

    # ---- Neural network (per motile) -----------------------------------------
    nn_inputs: int = 7
    nn_hidden: int = 6
    nn_outputs: int = 4

    # ---- Evolution -----------------------------------------------------------
    mutation_rate: float = 0.08
    resistance_mutation: float = 0.05
    selection_pressure: float = 0.05

    # ---- Infection -----------------------------------------------------------
    base_infection_prob: float = 0.12
    infection_weight_scramble: float = 0.25
    resistance_decay: float = 0.001
    infection_energy_drain: float = 0.3
    recovery_threshold: float = 0.02

    # ---- Output --------------------------------------------------------------
    save_every: int = 50
    output_dir: str = "states"


#: Default configuration singleton used when no overrides are needed.
DEFAULT_CONFIG = SimConfig()