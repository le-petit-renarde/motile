import unittest
import numpy as np
from framework import Brain, FeedForwardBrain, GridEnvironment

class TestFramework(unittest.TestCase):
    def test_brain(self):
        brain = FeedForwardBrain(layer_sizes=[7, 6, 4])
        self.assertEqual(len(brain.layer_sizes), 3)
        self.assertTrue(len(brain.weights) > 0)

        inputs = np.random.rand(7)
        outputs = brain.forward(inputs)
        self.assertEqual(outputs.shape, (4,))

        weights_before = brain.weights.copy()
        brain.mutate(0.1)
        self.assertFalse(np.array_equal(weights_before, brain.weights))

    def test_brain_base_mutate_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            Brain().mutate(0.1)

    def test_brain_mutate_requires_keyword_only_optional_args(self):
        brain = FeedForwardBrain(layer_sizes=[7, 6, 4])
        with self.assertRaises(TypeError):
            brain.mutate(0.1, 0.3, 0.01, 3.0)

    def test_environment(self):
        env = GridEnvironment(size=10, num_channels=2, diffusion_rates=[0.1, 0.2], decay_rates=[0.01, 0.02])
        self.assertEqual(env.grid.shape, (2, 10, 10, 10))

        # Add emission
        emissions = [(0, np.array([[5, 5, 5]]), np.array([1.0]))]
        env.step_chemistry(emissions)
        self.assertEqual(env.grid[0, 5, 5, 5], 1.0)

        env.step_chemistry()
        # Should diffuse and decay
        self.assertTrue(env.grid[0, 5, 5, 5] < 1.0)
        self.assertTrue(env.grid[0, 5, 6, 5] > 0.0)

    def test_environment_step_not_implemented(self):
        env = GridEnvironment(size=10, num_channels=1, diffusion_rates=[0.1], decay_rates=[0.01])
        with self.assertRaises(NotImplementedError):
            env.step([])

if __name__ == '__main__':
    unittest.main()
