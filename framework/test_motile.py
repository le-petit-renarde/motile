import unittest
from config import SimConfig
from motile import Motile
from engine import SimulationEngine

class TestMotile(unittest.TestCase):
    def test_motile_creation(self):
        config = SimConfig()
        m = Motile(config)
        self.assertIsNotNone(m)
        self.assertEqual(m.energy, 50.0)

    def test_engine_creation(self):
        config = SimConfig()
        e = SimulationEngine(config)
        e.reset()
        self.assertEqual(
            e.grid.shape,
            (3, config.grid_size, config.grid_size, config.grid_size),
        )

if __name__ == '__main__':
    unittest.main()
