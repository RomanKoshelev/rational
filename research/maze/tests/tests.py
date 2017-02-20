import unittest

import tensorflow as tf

from research.maze.ddpg.config import config
from research.maze.ddpg.ddpg_alg import DdpgAlgorithm
from research.maze.tests.logger import Logger
from research.maze.worlds.random_world import RandomWorld
from research.maze.worlds.target_world import TargetWorld


class Tests(unittest.TestCase):
    def test_random_world(self):
        log = Logger()
        log.subcribe()
        world = RandomWorld()
        alg = DdpgAlgorithm(config, tf.Session(), world)
        alg.train(10, 100)
        self.assertTrue(True)

    def test_target_world(self):
        log = Logger()
        log.subcribe()
        world = TargetWorld()
        config['alg.noise_rate_method'] = lambda _: 0.5
        alg = DdpgAlgorithm(config, tf.Session(), world)
        alg.train(3000, 30)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
