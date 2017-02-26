import unittest

from research.ddpg.test_ddpg import TestDdpg
from research.ddpg_per.config import config

config['train.episodes'] = 2000


class TestDddpgPer(TestDdpg):

    def test_default_config(self):
        self.run_experiment(config)

    def test_world_1d(self):
        config['world.dim'] = 1
        config['ddpg.buffer_size'] = 15 * 1000
        self.run_experiment(config)

    def test_world_2d(self):
        config['world.dim'] = 2
        config['ddpg.buffer_size'] = 15 * 1000
        self.run_experiment(config)


if __name__ == '__main__':
    unittest.main()
