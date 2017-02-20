import unittest

import tensorflow as tf

from research.maze.ddpg.ddpg_alg import DdpgAlgorithm
from research.maze.tests.config import config
from research.maze.tests.logger import Logger
from research.maze.worlds.random_world import RandomWorld
from research.maze.worlds.target_world import TargetWorld


class Test(unittest.TestCase):
    def test_random_world(self):
        log = Logger()
        log.subcribe()
        world = RandomWorld()
        alg = DdpgAlgorithm(config, tf.Session(), world)
        alg.train(10, 100)
        self.assertTrue(True)

    @staticmethod
    def run_experiment(cfg):
        log = Logger()
        log.subcribe()
        alg = DdpgAlgorithm(cfg, tf.Session(), TargetWorld(config))
        alg.train(cfg['train.episodes'], cfg['train.steps'])

    def test_target_world(self):
        # train
        config['train.episodes'] = 2000
        config['train.steps'] = 10
        # alg
        config['ddpg.buffer_size'] = 10*1000  # 10*1000
        config['ddpg.actor.lr'] = 1e-4  # 1e-4
        config['ddpg.critic.lr'] = 1e-3  # 1e-3

        config['ddpg.actor.tau'] = 1e-1  # 1e-3
        config['ddpg.critic.tau'] = 1e-1  # 1e-3

        self.run_experiment(config)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
