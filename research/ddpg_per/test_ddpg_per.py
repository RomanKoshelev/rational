import unittest

from research.ddpg.testddpg import TestDdpg
from research.ddpg_per.config import config


class TestDdpgPer(TestDdpg):

    def test_default_config(self):
        self.run_experiment(config)

    def test_world_1d(self):
        config['world.dim'] = 1
        config['ddpg.buffer_size'] = 15 * 1000
        self.run_experiment(config)

    def test_world_2d_best(self):
        config['world.dim'] = 2
        config['ddpg.buffer_size'] = 15 * 1000
        config['ddpg.actor_tau'] = 0.1  # 0.001
        config['ddpg.actor_lr'] = 1e-5  # 1e-4
        config['ddpg.critic_l2'] = 0.01  # 0.01
        config['ddpg.critic_tau'] = 0.01  # 0.001
        self.run_experiment(config)

    def test_world_2d(self):
        config['world.dim'] = 2
        self.run_experiment(config)

    def test_world_3d(self):
        config['world.dim'] = 3
        config['per.degree'] = .9  # 0.6
        self.run_experiment(config)

    def test_world_4d(self):
        config['world.dim'] = 4
        self.run_experiment(config)


if __name__ == '__main__':
    unittest.main()
