#!/usr/bin/env python3
"""Motile — 3D Artificial Life Simulation with Host-Phage Neuroevolution.
Usage:
    python run.py [--ticks 10000] [--grid 48] [--pop 150] [--save-every 50] [--no-gui]
"""

import argparse
import json
import os
import sys
import time
import numpy as np

from config import SimConfig
from engine import SimulationEngine
from motile import Motile, MotileBrain
from phage import PhageSystem
from evolution import EvolutionManager


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars and arrays to native Python types."""
    def default(self, obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _json_dump(obj, fp):
    """Helper: dump with numpy-aware encoder."""
    json.dump(obj, fp, indent=1, cls=NumpyEncoder)


def parse_args():
    p = argparse.ArgumentParser(description='Motile — 3D Artificial Life Simulation')
    p.add_argument('--ticks', type=int, default=10000, help='Number of simulation ticks')
    p.add_argument('--grid', type=int, default=48, help='Grid size (N×N×N)')
    p.add_argument('--pop', type=int, default=150, help='Initial population')
    p.add_argument('--save-every', type=int, default=50, help='Save state every N ticks')
    p.add_argument('--seed', type=int, default=None, help='Random seed')
    p.add_argument('--output', type=str, default='states', help='Output directory')
    p.add_argument('--quiet', action='store_true', help='Suppress progress output')
    return p.parse_args()


def run_simulation(config: SimConfig, num_ticks: int, save_every: int, output_dir: str, quiet: bool = False):
    """Main simulation loop."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize systems
    engine = SimulationEngine(config)
    engine.reset()
    phage_sys = PhageSystem(config)
    evo = EvolutionManager(config)
    
    # Create initial population with random positions and brains
    motiles = []
    for i in range(config.initial_population):
        motiles.append(Motile(config, generation=0))
    
    # Seed initial infection: infect ~5% of starting population
    np.random.shuffle(motiles)
    seed_count = max(5, int(config.initial_population * 0.20))
    for i in range(seed_count):
        motiles[i].status = 'INFECTED'
        motiles[i].brain.scramble(0.1)  # mild initial disruption
    
    history = []
    start_time = time.time()
    pop_stats = {}
    
    for tick in range(num_ticks):
        # 1. Sense & decide (all motiles) — vectorized
        inputs_list = []
        
        # Precompute all grid positions for neighbor density
        all_positions = np.array([m.grid_pos() for m in motiles], dtype=int)
        n = len(all_positions)

        # Vectorized pairwise Chebyshev distances
        neighbor_counts = np.zeros(n, dtype=int)
        if n > 0:
            # Compute all pairwise Chebyshev distances max(|x1-x2|, |y1-y2|, |z1-z2|)
            diffs = np.abs(all_positions[:, None, :] - all_positions[None, :, :])
            chebyshev_dist = diffs.max(axis=2)
            # Count neighbors within Chebyshev distance 3 (subtract 1 for self)
            neighbor_counts = (chebyshev_dist <= 3).sum(axis=1) - 1
            neighbor_counts = np.maximum(0, neighbor_counts)
        
        for i, m in enumerate(motiles):
            gp = all_positions[i]
            neighbor_density = min(neighbor_counts[i] / 20.0, 1.0)
            
            # Build input with actual neighbor density
            nutrient, toxin, phage = engine.sample_at(gp.reshape(1, 3))
            grad = engine.gradient_at(gp.reshape(1, 3), engine.NUTRIENT)
            
            inp = np.array([
                np.clip(grad[0, 0] * 5, -1, 1),
                np.clip(grad[0, 1] * 5, -1, 1),
                np.clip(grad[0, 2] * 5, -1, 1),
                np.clip(toxin[0] * 2, -1, 1),
                np.clip(phage[0] * 2, -1, 1),
                np.clip(m.energy / 100.0 - 0.5, -1, 1),
                np.clip(neighbor_density * 2, -1, 1),
            ], dtype=np.float32)
            inputs_list.append(inp)
        
        # Batch forward pass for efficiency
        if inputs_list:
            batch = np.stack(inputs_list)
            outputs = np.array([m.decide(inp) for m, inp in zip(motiles, inputs_list)])
        else:
            outputs = np.zeros((0, 4))
        
        # 2. Apply movement
        positions_arr = np.array([m.position for m in motiles])
        velocities = outputs[:, :3] * config.max_speed
        positions_arr += velocities
        
        # Boundary: soft bounce (reflect with damping)
        for dim in range(3):
            below = positions_arr[:, dim] < 0
            above = positions_arr[:, dim] > config.grid_size - 1
            positions_arr[below, dim] = -positions_arr[below, dim] * 0.3
            positions_arr[above, dim] = 2 * (config.grid_size - 1) - positions_arr[above, dim]
            positions_arr[above, dim] = np.clip(positions_arr[above, dim], 0, config.grid_size - 1)
        positions_arr = np.clip(positions_arr, 0, config.grid_size - 1)
        
        for i, m in enumerate(motiles):
            m.position = positions_arr[i]
            m.velocity = velocities[i]
        
        # 3. Phage infection tick
        stats, emit_positions, phage_emissions = phage_sys.tick(motiles, engine)
        
        # 4. Metabolism & emissions
        toxin_emissions = np.array([m.emit_toxin() for m in motiles], dtype=np.float32)
        
        # 5. Chemistry tick
        engine.tick_chemistry(
            motile_positions=np.array([m.grid_pos() for m in motiles], dtype=int),
            motile_toxin_emit=toxin_emissions,
            motile_phage_emit=phage_emissions,
        )
        
        # 6. Evolution tick
        evo_stats = evo.tick(motiles)
        
        # 7. Kill motiles that starved
        alive = [m for m in motiles if m.metabolize()]
        motiles[:] = alive
        
        # 8. Nutrient consumption (motiles at positions with high nutrient consume it)
        for m in motiles:
            gp = m.grid_pos()
            nutrient_here = engine.grid[engine.NUTRIENT, gp[0], gp[1], gp[2]]
            consume = min(nutrient_here * 0.15, 0.05)
            engine.grid[engine.NUTRIENT, gp[0], gp[1], gp[2]] -= consume
            m.energy += consume * 15  # convert nutrient to energy
        
        # 9. Save snapshot
        if tick % save_every == 0 and len(motiles) > 0:
            pop_stats = evo.population_stats(motiles)
            snapshot = {
                'tick': tick,
                'population': len(motiles),
                'stats': pop_stats,
                'infection': phage_sys.stats,
                'chemistry': engine.get_snapshot(),
                'motiles': [m.to_dict() for m in motiles],
                'grid_slice': engine.grid[:, :, config.grid_size//2, :].tolist(),  # Y-mid slice for 2D view
            }
            filepath = os.path.join(output_dir, f'state_{tick:06d}.json')
            with open(filepath, 'w') as f:
                _json_dump(snapshot, f)
            history.append(snapshot)
        
        # 10. Progress
        if not quiet and tick % 500 == 0:
            elapsed = time.time() - start_time
            tps = tick / max(elapsed, 0.001)
            pop = len(motiles)
            infected = phage_sys.stats['total_infected']
            print(f"Tick {tick:6d} | Pop {pop:3d} | Infected {infected:3d} | "
                  f"AvgResist {pop_stats.get('avg_resistance',0):.3f} | "
                  f"TPS {tps:6.1f}")
        
        # Abort if population extinct
        if len(motiles) == 0:
            if not quiet:
                print(f"\nPopulation extinct at tick {tick}!")
            break
    
    # Final save
    if motiles:
        pop_stats = evo.population_stats(motiles)
        snapshot = {
            'tick': num_ticks,
            'population': len(motiles),
            'stats': pop_stats,
            'infection': phage_sys.stats,
            'chemistry': engine.get_snapshot(),
            'motiles': [m.to_dict() for m in motiles],
            'grid_slice': engine.grid[:, :, config.grid_size//2, :].tolist(),
        }
        with open(os.path.join(output_dir, 'state_final.json'), 'w') as f:
            _json_dump(snapshot, f)
    
    elapsed = time.time() - start_time
    if not quiet:
        print(f"\nDone: {num_ticks} ticks in {elapsed:.1f}s ({num_ticks/elapsed:.1f} tps)")
        print(f"Final population: {len(motiles)}")
        print(f"States saved to: {output_dir}/")
    
    return history


if __name__ == '__main__':
    args = parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
    
    config = SimConfig(
        grid_size=args.grid,
        initial_population=args.pop,
        save_every=args.save_every,
        output_dir=args.output,
    )
    
    print(f"Motile — 3D Artificial Life Simulation")
    print(f"Grid: {config.grid_size}³ | Initial pop: {config.initial_population}")
    print(f"Ticks: {args.ticks} | Save every: {args.save_every}")
    print(f"Output: {args.output}/\n")
    
    history = run_simulation(config, args.ticks, args.save_every, args.output, args.quiet)