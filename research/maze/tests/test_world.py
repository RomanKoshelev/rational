import unittest

import tensorflow as tf

from research.maze.ddpg.config import config
from research.maze.ddpg.ddpg_alg import DdpgAlgorithm
from research.maze.tests.logger import Logger
from research.maze.tests.random_world import RandomWorld
from research.maze.tests.simple_world import TargetWorld


class EventTests(unittest.TestCase):
    def test_random_world(self):
        log = Logger()
        log.subcribe()
        world = RandomWorld()
        alg = DdpgAlgorithm(config, tf.Session(), world)
        alg.train(10, 100)
        self.assertTrue(True)

    def test_simple_world(self):
        log = Logger()
        log.subcribe()
        world = TargetWorld()
        alg = DdpgAlgorithm(config, tf.Session(), world)
        alg.train(1000, 20)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
