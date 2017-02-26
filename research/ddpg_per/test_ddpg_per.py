import unittest

from research.ddpg.test_ddpg import TestDdpg
from research.ddpg_per.config import config


class TestDdpgPer(TestDdpg):

    def test_default_config(self):
        self.run_experiment(config)

    def test_world_1d(self):
        config['world.dim'] = 1
        self.run_experiment(config)

    def test_world_2d(self):
        config['world.dim'] = 2
        self.run_experiment(config)

    def test_world_3d(self):
        config['world.dim'] = 3
        # config['per.degree'] = 1.  # 0.6
        self.run_experiment(config)

    def test_world_4d(self):
        config['world.dim'] = 4
        # config['per.degree'] = 1.1  # 0.6
        self.run_experiment(config)

    def test_world_5d(self):
        config['world.dim'] = 5
        # config['per.degree'] = 1.2  # 0.6
        self.run_experiment(config)

    def test_world_10d(self):
        config['world.dim'] = 10
        config['train.episodes'] = 3000
        config['ddpg.buffer_size'] = 50 * 1000
        config['per.degree'] = 1.2  # 0.6
        self.run_experiment(config)


if __name__ == '__main__':
    unittest.main()
