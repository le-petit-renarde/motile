"""Phage infection lifecycle: spread, infection, resistance, recovery.

Manages the full epidemiological cycle for motile agents: phage emission from
infected hosts, environmental exposure via local-grid density checks, infection
rolls with resistance-modulated probability, brain-weight scrambling on infection,
stochastic recovery into a RESISTANT state with acquired-immunity boost, and slow
resistance decay enabling epidemic resurgence.

Typical usage:
    phage_sys = PhageSystem(config)
    for tick in range(N):
        stats, positions, emissions = phage_sys.tick(motiles, engine)
        engine.deposit_phage(positions, emissions)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from config import DEFAULT_CONFIG, SimConfig

# ---------------------------------------------------------------------------
# Constants (tunable, validated by clamp helpers below)
# ---------------------------------------------------------------------------
_RESISTANCE_INFECTION_SCALE = 0.8   # how much motile.resistance reduces infection prob
_RESISTANCE_RECOVERY_SCALE   = 5.0   # how much motile.resistance boosts recovery prob
_RECOVERY_TICK_BONUS_CAP     = 0.1   # cap on the "survived longer" recovery bonus
_RECOVERY_TICK_BONUS_SLOPE   = 0.001 # per-tick bonus slope
_RESISTANCE_GAIN             = 0.15  # acquired resistance on recovery
_MIN_RESISTANCE              = 0.05  # resistance floor during decay
_PEAK_SIGNIFICANCE           = 3     # minimum infected count for a wave peak


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clamp_probability(value: float) -> float:
    """Clamp a probability-like value into the valid [0.0, 1.0] range."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _count_status(motiles: List, status: str) -> int:
    """Count motiles whose `.status` equals *status*."""
    return sum(1 for m in motiles if m.status == status)


# ---------------------------------------------------------------------------
# PhageSystem
# ---------------------------------------------------------------------------
class PhageSystem:
    """Manages infection spread through the motile population.

    Each *tick* the system performs three phases:

    1. **Emission** – every infected motile emits phage particles into the
       surrounding spatial grid (delegated to the engine).
    2. **Infection** – each HEALTHY motile samples the local phage density;
       if it exceeds ``phage_infection_threshold`` an infection roll is made,
       modulated by the motile's innate resistance.
    3. **Recovery** – each INFECTED motile may recover (transition to
       RESISTANT), gaining a permanent resistance bump.  RESISTANT motiles
       lose a small amount of resistance each tick, allowing phage to
       eventually return.

    Attributes
    ----------
    config : SimConfig
        Active simulation configuration.
    rng : numpy.random.RandomState
        Isolated random state for reproducibility.
    stats : dict
        Most-recent tick statistics (infections, recoveries, totals).
    """

    def __init__(self, config: SimConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.rng = np.random.RandomState()  # noqa: NPY002 – legacy state for compat

        # Statistics for visualization -------------------------------------------------
        self.stats: Dict[str, int] = {
            "infections_this_tick": 0,
            "recoveries_this_tick": 0,
            "total_infected": 0,
            "total_resistant": 0,
            "total_healthy": 0,
            "total": 0,
        }

    # ------------------------------------------------------------------ public API
    def tick(
        self,
        motiles: List,
        engine,
    ) -> Tuple[Dict[str, int], np.ndarray, np.ndarray]:
        """Advance the phage system by one simulation tick.

        Parameters
        ----------
        motiles : list of Motile
            Every motile currently alive in the simulation.
        engine :
            The simulation engine.  Expected to expose:
            ``engine.grid`` with a ``PHAGE`` layer and
            ``engine.grid[engine.PHAGE, x, y, z]`` access.

        Returns
        -------
        stats : dict
            Tick-level statistics (keys: ``infections_this_tick``,
            ``recoveries_this_tick``, ``total_infected``,
            ``total_resistant``, ``total_healthy``, ``total``).
        positions : np.ndarray
            Integer array of shape ``(N, 3)`` holding each motile's grid
            position (for upstream phage deposition).
        phage_emissions : np.ndarray
            Float array of shape ``(N,)`` with the phage quantity emitted
            by each motile this tick.
        """
        cfg = self.config
        n = len(motiles)

        if n == 0:
            self.stats = {
                "infections_this_tick": 0,
                "recoveries_this_tick": 0,
                "total_infected": 0,
                "total_resistant": 0,
                "total_healthy": 0,
                "total": 0,
            }
            return self.stats, np.empty((0, 3), dtype=int), np.empty((0,), dtype=np.float32)

        # --- collect emissions ---------------------------------------------------
        positions_float = np.array([m.position for m in motiles])
        positions = np.maximum(0, np.minimum(cfg.grid_size - 1, np.rint(positions_float).astype(int)))
        phage_emissions = np.array([m.emit_phage() for m in motiles], dtype=np.float32)

        # --- per-motile infection / recovery loop --------------------------------
        infections = 0
        recoveries = 0

        for i, motile in enumerate(motiles):
            gp = positions[i]
            # Sample phage in 3×3×3 neighborhood around motile
            x0, y0, z0 = max(0, gp[0]-1), max(0, gp[1]-1), max(0, gp[2]-1)
            x1, y1, z1 = min(cfg.grid_size, gp[0]+2), min(cfg.grid_size, gp[1]+2), min(cfg.grid_size, gp[2]+2)
            phage_level = engine.grid[engine.PHAGE, x0:x1, y0:y1, z0:z1].max()

            if motile.status == "HEALTHY":
                if phage_level > cfg.phage_infection_threshold:
                    infection_chance = cfg.base_infection_prob * (
                        1.0 - motile.resistance * _RESISTANCE_INFECTION_SCALE
                    )
                    infection_chance = _clamp_probability(infection_chance)

                    if self.rng.random() < infection_chance:
                        motile.status = "INFECTED"
                        motile.infection_ticks = 0
                        # Behavioural disruption: scramble brain weights
                        motile.brain.scramble(cfg.infection_weight_scramble)
                        infections += 1

            elif motile.status == "INFECTED":
                # Recovery: base chance + resistance bonus + survival-time bonus
                recovery_chance = cfg.recovery_threshold * (
                    1.0 + motile.resistance * _RESISTANCE_RECOVERY_SCALE
                )
                recovery_chance += min(
                    motile.infection_ticks * _RECOVERY_TICK_BONUS_SLOPE,
                    _RECOVERY_TICK_BONUS_CAP,
                )
                recovery_chance = _clamp_probability(recovery_chance)

                if self.rng.random() < recovery_chance:
                    motile.status = "RESISTANT"
                    # Acquired immunity bump
                    motile.resistance = min(1.0, motile.resistance + _RESISTANCE_GAIN)
                    recoveries += 1

            elif motile.status == "RESISTANT":
                # Imperfect immunity: resistance slowly decays over time
                motile.resistance = max(_MIN_RESISTANCE, motile.resistance - cfg.resistance_decay)

        # --- update statistics snapshot ------------------------------------------
        self.stats = {
            "infections_this_tick": infections,
            "recoveries_this_tick": recoveries,
            "total_infected": _count_status(motiles, "INFECTED"),
            "total_resistant": _count_status(motiles, "RESISTANT"),
            "total_healthy": _count_status(motiles, "HEALTHY"),
            "total": n,
        }

        return self.stats, positions, phage_emissions


# ---------------------------------------------------------------------------
# Epidemiological analysis
# ---------------------------------------------------------------------------
def infection_wave_analysis(history: List[dict]) -> dict:
    """Analyse infection-wave patterns from a simulation-history trace.

    Detects epidemic peaks, estimates the average inter-peak period, and
    computes coarse resistance / infectivity trend lines over the run.

    Parameters
    ----------
    history : list of dict
        Per-tick records.  Each dict is expected to contain:
        - ``stats``: a dict with key ``total_infected``
        - ``avg_resistance``: scalar float
        - ``avg_infectivity``: scalar float

    Returns
    -------
    dict
        Keys:
        - ``waves``: number of detected epidemic peaks
        - ``avg_period``: mean tick-gap between peaks (0 if ≤ 1 peak)
        - ``resistance_trend``: difference (last − first) in avg. resistance
        - ``infectivity_trend``: difference (last − first) in avg. infectivity
    """
    if len(history) < 2:
        return {
            "waves": 0,
            "avg_period": 0.0,
            "resistance_trend": 0.0,
            "infectivity_trend": 0.0,
        }

    infected_counts = np.array([h["stats"]["total_infected"] for h in history], dtype=int)
    resistance_avgs = np.array([h.get("avg_resistance", 0.0) for h in history], dtype=np.float64)
    infectivity_avgs = np.array([h.get("avg_infectivity", 0.0) for h in history], dtype=np.float64)

    # Simple peak detection on infected counts
    peaks: List[int] = []
    for i in range(1, len(infected_counts) - 1):
        lo = infected_counts[i - 1]
        hi = infected_counts[i + 1]
        cur = infected_counts[i]
        if cur > lo and cur > hi and cur > _PEAK_SIGNIFICANCE:
            peaks.append(i)

    diffs = np.diff(np.array(peaks, dtype=np.float64))
    avg_period = float(np.mean(diffs)) if len(diffs) > 0 else 0.0

    return {
        "waves": len(peaks),
        "avg_period": avg_period,
        "resistance_trend": float(resistance_avgs[-1] - resistance_avgs[0])
        if len(resistance_avgs) > 0
        else 0.0,
        "infectivity_trend": float(infectivity_avgs[-1] - infectivity_avgs[0])
        if len(infectivity_avgs) > 0
        else 0.0,
    }