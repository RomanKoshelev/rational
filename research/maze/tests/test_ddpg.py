import unittest

import tensorflow as tf

from common.events import EventSystem
from common.text_utils import fields
from research.maze.ddpg.ddpg_alg import DdpgAlgorithm
from research.maze.tests.config import config
from research.maze.tests.logger import Logger
from research.maze.tests.timer import Timer
from research.maze.worlds.random_world import RandomWorld
from research.maze.worlds.target_world import TargetWorld


class TestDdpg(unittest.TestCase):
    def test_random_world(self):
        Logger()
        alg = DdpgAlgorithm(config, RandomWorld())
        alg.train(10, 100)
        self.assertTrue(True)

    def run_experiment(self, cfg):
        with Timer(), Logger(), tf.Session():
            r, d = None, None
            episodes, steps = cfg['train.episodes'], cfg['train.steps']

            def on_train_episode(_):
                nonlocal r, d
                r, d = alg.eval(10, steps)

            EventSystem.subscribe('algorithm.train_episode', on_train_episode)
            alg = DdpgAlgorithm(cfg, TargetWorld(config))
            alg.train(episodes, steps)

            EventSystem.send('log', ["-" * 32, fields([['reward', r], ['done', d]], -6)])
            self.assertGreater(r, 900)
            self.assertGreater(d, .75)

    def test_default_config(self):
        self.run_experiment(config)

    def test_buffer_size(self):
        config['train.buffer_size'] = 2 * 1000  # 10 * 1000
        self.run_experiment(config)

    def test_batch_size(self):
        config['train.buffer_size'] = 2 * 1000  # 10 * 1000
        config['ddpg.batch_size'] = 256  # 128
        self.run_experiment(config)


if __name__ == '__main__':
    unittest.main()
