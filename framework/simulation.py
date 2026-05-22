import numpy as np
from typing import List, Optional
from .base import Simulation, Agent, Environment

class BaseSimulation(Simulation):
    """
    An extensible simulation loop. It separates agent perception/action,
    environment step, and population dynamics (birth/death).
    """
    def __init__(self, env: Environment):
        super().__init__(env)
        self.stats = {}

    def _step_agents(self) -> None:
        """
        Batches agent observations, runs neural networks, and applies actions.
        """
        if not self.agents:
            return

        inputs_list = []
        for agent in self.agents:
            obs = agent.sense(self.env)
            inputs_list.append(obs)

        # Optional: batch forward pass if using a uniform brain architecture
        # For full generality, we process individually:
        outputs = []
        for agent, obs in zip(self.agents, inputs_list):
            if agent.brain:
                out = agent.brain.forward(obs)
            else:
                out = np.zeros(0) # dummy
            outputs.append(out)

        for agent, action in zip(self.agents, outputs):
            agent.act(action, self.env)

    def _step_population(self) -> None:
        """
        Override this to handle reproduction, mutations, and death.
        """
        pass

    def run(self, num_ticks: int, quiet: bool = False) -> None:
        """Run the simulation for a number of ticks."""
        for _ in range(num_ticks):
            self.step()
            if not self.agents:
                if not quiet:
                    print(f"Population extinct at tick {self.tick}!")
                break
