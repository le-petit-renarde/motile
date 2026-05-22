"""Population-level evolution: reproduction, selection, population management."""

import numpy as np
from typing import List
from config import SimConfig, DEFAULT_CONFIG

class EvolutionManager:
    """Handles reproduction, natural selection, and population control."""
    
    def __init__(self, config: SimConfig = DEFAULT_CONFIG):
        self.config = config
        self.rng = np.random.RandomState()
        self.generation = 0
        self.total_births = 0
        self.total_deaths = 0
    
    def tick(self, motiles: List) -> dict:
        """
        One evolution tick:
        1. Check which motiles can reproduce → create children
        2. Remove dead motiles (energy <= 0)
        3. Apply selection pressure if overpopulated
        
        Returns stats dict.
        """
        cfg = self.config
        births = 0
        deaths_by_energy = 0
        deaths_by_selection = 0
        
        # Reproduction
        children = []
        for motile in motiles:
            if motile.can_divide():
                # Density-dependent reproduction: fewer children in crowded areas
                # (handled by simulation loop, here just probabilistic)
                if self.rng.random() < 0.02:  # ~2% chance per tick
                    child = motile.create_child()
                    children.append(child)
                    births += 1
        
        # Add children to population
        motiles.extend(children)
        
        # Remove dead (energy ≤ 0)
        alive = [m for m in motiles if m.energy > 0]
        deaths_by_energy = len(motiles) - len(alive)
        motiles[:] = alive
        
        # Selection pressure: if overpopulated, kill weakest
        if len(motiles) > cfg.max_population:
            excess = len(motiles) - cfg.max_population
            # Sort by energy (ascending) and kill the weakest
            motiles.sort(key=lambda m: m.energy)
            # But preferentially kill INFECTED motiles first (disease culling)
            infected_first = sorted(motiles, key=lambda m: (0 if m.status == 'INFECTED' else 1, m.energy))
            victims = infected_first[:excess]
            for v in victims:
                motiles.remove(v)
            deaths_by_selection = len(victims)
        
        self.total_births += births
        self.total_deaths += deaths_by_energy + deaths_by_selection
        
        return {
            'births': births,
            'deaths_energy': deaths_by_energy,
            'deaths_selection': deaths_by_selection,
            'population': len(motiles),
            'total_births': self.total_births,
            'total_deaths': self.total_deaths,
        }
    
    def population_stats(self, motiles: List) -> dict:
        """Compute summary statistics for the population."""
        if not motiles:
            return {'avg_energy': 0, 'avg_resistance': 0, 'avg_infectivity': 0,
                    'avg_age': 0, 'avg_generation': 0, 'healthy': 0, 'infected': 0, 'resistant': 0}
        
        energies = [m.energy for m in motiles]
        resistances = [m.resistance for m in motiles]
        infectivities = [m.infectivity for m in motiles]
        ages = [m.age for m in motiles]
        gens = [m.generation for m in motiles]
        statuses = [m.status for m in motiles]
        
        return {
            'avg_energy': round(float(np.mean(energies)), 2),
            'avg_resistance': round(float(np.mean(resistances)), 4),
            'avg_infectivity': round(float(np.mean(infectivities)), 4),
            'avg_age': round(float(np.mean(ages)), 2),
            'avg_generation': round(float(np.mean(gens)), 2),
            'max_generation': max(gens),
            'healthy': statuses.count('HEALTHY'),
            'infected': statuses.count('INFECTED'),
            'resistant': statuses.count('RESISTANT'),
        }