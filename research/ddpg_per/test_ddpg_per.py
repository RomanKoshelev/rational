import unittest

from research.ddpg.test_ddpg import TestDdpg
from research.ddpg_per.config import config


class TestDddpgPer(TestDdpg):

    def test_default_config(self):
        self.run_experiment(config)

    def test_world_2d(self):
        config['world.dim'] = 2
        config['ddpg.buffer_size'] = 10 * 1000
        self.run_experiment(config)


if __name__ == '__main__':
    unittest.main()
