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

    def test_target_world(self):
        # world
        config['world.done_dist'] = .5
        # alg
        config['alg.buffer_size'] = 10*1000
        config['alg.actor.lr'] = 1e-4
        config['alg.critic.lr'] = 1e-3
        log = Logger()
        log.subcribe()
        alg = DdpgAlgorithm(config, tf.Session(), TargetWorld(config))
        alg.train(2000, 10)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
