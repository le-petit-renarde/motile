import numpy as np
from typing import Optional

class Brain:
    """Base class for an agent's neural network brain."""
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def mutate(self, mutation_rate: float, leap_prob: float, leap_multiplier: float) -> None:
        """Mutate the brain's parameters."""
        pass

    def copy(self) -> 'Brain':
        """Return a copy of the brain."""
        raise NotImplementedError

class Agent:
    """Base class for a simulated agent."""
    __slots__ = ("id", "position", "energy", "age", "generation", "brain")

    _next_id = 0

    def __init__(self, position: Optional[np.ndarray] = None, brain: Optional[Brain] = None, energy: float = 0.0, generation: int = 0):
        self.id = Agent._next_id
        Agent._next_id += 1
        self.position = position if position is not None else np.zeros(3, dtype=np.float32)
        self.brain = brain
        self.energy = energy
        self.age = 0
        self.generation = generation

    def sense(self, env) -> np.ndarray:
        """Construct the sensory input vector."""
        raise NotImplementedError

    def act(self, action: np.ndarray, env) -> None:
        """Apply the action chosen by the brain."""
        raise NotImplementedError

class Environment:
    """Base class for the simulation environment."""
    def step(self, agents: list[Agent]) -> None:
        """Advance the environment state."""
        raise NotImplementedError

class Simulation:
    """Base class for the main simulation loop."""
    def __init__(self, env: Environment):
        self.env = env
        self.agents: list[Agent] = []
        self.tick = 0

    def add_agent(self, agent: Agent) -> None:
        self.agents.append(agent)

    def step(self) -> None:
        """Run one full tick of the simulation."""
        self._step_agents()
        self.env.step(self.agents)
        self._step_population()
        self.tick += 1

    def _step_agents(self) -> None:
        """Let all agents sense and act."""
        for agent in self.agents:
            obs = agent.sense(self.env)
            if agent.brain:
                action = agent.brain.forward(obs)
            else:
                action = np.zeros(0) # or dummy
            agent.act(action, self.env)

    def _step_population(self) -> None:
        """Handle reproduction and death."""
        raise NotImplementedError
