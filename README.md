# 🧬 Motile — 3D Artificial Life with Host-Phage Neuroevolution

**Motile** is a 3D artificial life simulation where tiny organisms evolve neural-network brains, battle viral infections, and develop resistance in a real-time chemical ecosystem.

Inspired by *Creatures* (1996) but scaled to the microbial: microscopic motiles swim through a 3D nutrient field, each with a tiny 7→6→4 neural network that IS its genome. No backprop, no training — behaviors emerge through generational evolution and selection pressure. A phage epidemic burns through the population, scrambling brains and driving an evolutionary arms race.

## Quick Start

```bash
# Install (only needs numpy)
pip install numpy

# Run a simulation
python run.py --grid 16 --pop 100 --ticks 2000 --save-every 100

# Open the viewer (no server needed — just open in browser)
open viewer.html
```

Drop the generated `demo/state_*.json` files (or the whole folder) onto the viewer page.

## How It Works

### The Simulation

```
┌──────────────────────────────────────┐
│  N×N×N 3D Grid                       │
│                                      │
│  Chemical fields (convolve + diffuse)│
│  • Nutrient — flows in from top      │
│  • Toxin   — metabolic waste         │
│  • Phage   — infectious particles    │
│                                      │
│  Motiles (50–350 organisms)          │
│  ○ Position, velocity, energy        │
│  ○ Tiny NN brain (7→6→4, ~50 weights)│
│  ○ Genome = NN weights (evolvable)   │
│  ○ Status: HEALTHY|INFECTED|RESISTANT│
└──────────────────────────────────────┘
```

### The Neural Network (per organism)

```
Input (7):                    Hidden (6)   Output (4):
──────────────                ──────────   ──────────────
Nutrient gradient X ─┐                    Movement ΔX ──►
Nutrient gradient Y ─┤                    Movement ΔY ──►
Nutrient gradient Z ─┤   tanh → tanh →    Movement ΔZ ──►
Toxin level ─────────┤                    Action ───────►
Phage level ─────────┤
Energy state ────────┤
Neighbor density ────┘
```

### The Infection Cycle

```
HEALTHY ──(phage exposure)──► INFECTED ──(survive+immune)──► RESISTANT
   ▲                              │                               │
   │                              │ brain scrambled               │
   │                              │ emits phage particles         │ resistance
   │                              │ energy drain ↑↑              │ decays over
   │                              ▼                               │ time
   └────────(new generation)────── ◄──────────────────────────────┘
```

### Reproduction & Evolution

- Motile splits when energy ≥ 80 → parent pays 50, child gets 30
- Child inherits parent's NN weights + Gaussian mutation (σ = 0.08)
- Occasional "leap" mutations (1% chance, 3× magnitude)
- Resistant parents pass immunity bonus to children
- When population hits cap (350), weakest motiles are culled

### Emergent Behaviors

- **Chemotaxis** — motiles learn to swim toward nutrient gradients
- **Phage waves** — epidemics surge, crash, resurge as resistance cycles
- **Resistance arms race** — population resistance rises during outbreaks, dilutes during calm periods
- **Spatial avoidance** — some lineages evolve to avoid phage-dense regions
- **Coexistence** — susceptible fast-breeders and resistant slow-breeders share the volume

## Files

| File | Purpose |
|---|---|
| `run.py` | CLI entry point |
| `config.py` | All tunable parameters (dataclass) |
| `engine.py` | 3D grid, chemical diffusion (pure numpy) |
| `motile.py` | Organism class + neural network |
| `phage.py` | Infection spread, recovery, immunity |
| `evolution.py` | Reproduction, selection, population stats |
| `viewer.html` | Standalone Three.js 3D viewer |

## CLI Options

```
python run.py [OPTIONS]

--grid N        Grid size (N×N×N), default 48
--pop N         Initial population, default 150
--ticks N       Simulation ticks, default 10000
--save-every N  Save state every N ticks, default 50
--seed N        Random seed for reproducibility
--output DIR    Output directory for state snapshots
--quiet         Suppress progress output
```

## Viewer Controls

- **Drag & drop** state JSON files onto the viewer
- **Space** — play/pause animation
- **← →** — previous/next frame
- **0** — reset camera
- **Mouse** — orbit, zoom, pan
- Timeline scrubber at bottom

Colors:
- 🟢 Green = Healthy
- 🔴 Red (pulsing) = Infected
- 🔵 Blue = Resistant
- ⚪ White haze = Phage cloud

## Requirements

- Python 3.10+ with numpy
- A modern browser with WebGL (for viewer)
- No GPU, no ML frameworks, no database

## Tuning for Infection Dynamics

If infections die out too fast:
- Lower `phage_infection_threshold` (easier to catch)
- Lower `recovery_threshold` (stay infected longer)
- Increase `phage_emit_rate` (more viral shedding)
- Increase `base_infection_prob` (higher transmission chance)

If everyone dies from infection:
- Lower `infection_energy_drain` (sick but not starving)
- Increase `recovery_threshold` (faster recovery)
- Lower `base_infection_prob` (harder to transmit)

## Inspired By

- **Creatures** (1996) — Steve Grand's artificial life with neural norns
- **Conway's Game of Life** — emergent complexity from simple rules
- **NEAT** — neuroevolution of augmenting topologies
- Real microbial chemotaxis and phage therapy research

## License

MIT
