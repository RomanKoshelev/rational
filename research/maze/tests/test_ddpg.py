import unittest

import tensorflow as tf

from common.events import Events
from research.maze.ddpg.ddpg_alg import DdpgAlgorithm
from research.maze.tests.config import config
from research.maze.tests.logger import Logger
from research.maze.worlds.random_world import RandomWorld
from research.maze.worlds.target_world import TargetWorld


class TestDdpg(unittest.TestCase):
    def test_random_world(self):
        Logger()
        alg = DdpgAlgorithm(config, RandomWorld())
        alg.train(10, 100)
        self.assertTrue(True)

    @staticmethod
    def run_experiment(cfg):
        Logger()
        r, q = None, None

        episodes, steps = cfg['train.episodes'], cfg['train.steps']

        def on_train_episode(data):
            nonlocal r, q
            e = data['episode']
            r = data['reward']
            q = data['qmax']
            if e % 10 == 0:
                alg.eval(1, steps)

        with tf.Session():
            Events.subscribe('algorithm.train_episode', on_train_episode)
            alg = DdpgAlgorithm(cfg, TargetWorld(config))
            alg.train(episodes, steps)
        return r, q

    def test_default_config(self):
        r, q = self.run_experiment(config)
        self.assertGreater(r, 90)
        self.assertGreater(q, 800)

    def test_buffer_size(self):
        config['train.buffer_size'] = 2*1000  # 10 * 1000
        r, q = self.run_experiment(config)
        self.assertGreater(r, 90)
        self.assertGreater(q, 800)

    def test_batch_size(self):
        config['train.episodes'] = 2000
        config['train.buffer_size'] = 2*1000  # 10 * 1000
        config['ddpg.batch_size'] = 1024  # 256! 128
        r, q = self.run_experiment(config)
        self.assertGreater(r, 90)
        self.assertGreater(q, 800)


if __name__ == '__main__':
    unittest.main()
