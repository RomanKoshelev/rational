import unittest

import tensorflow as tf

from common.events import EventSystem
from common.text_utils import fields
from research.maze.ddpg.ddpg_alg import DdpgAlgorithm
from research.maze.tests.config import config
from research.maze.tests.logger import TrainLogger
from research.maze.tests.timer import Timer


class TestDdpg(unittest.TestCase):
    def run_experiment(self, cfg):
        with Timer(), TrainLogger(), tf.Session():
            r, d = 0, 0
            episodes, steps = cfg['train.episodes'], cfg['train.steps']

            def on_train_episode(_):
                nonlocal r, d
                r, d = alg.eval(10, steps)

            EventSystem.subscribe('algorithm.train_episode', on_train_episode)
            alg = DdpgAlgorithm(cfg, config['world.class'](config))
            alg.train(episodes, steps)

            EventSystem.send('train.summary', ["\n", "-" * 32, fields([
                ['Reward', "%.2f" % r],
                ['Done', "%.0f%%" % (d*100)]
            ], -6)])
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

    def test_world_1d(self):
        config['world.dim'] = 1
        self.run_experiment(config)

    def test_world_2d(self):
        config['world.dim'] = 2
        self.run_experiment(config)

    def test_world_3d(self):
        config['world.dim'] = 3
        self.run_experiment(config)

    def test_world_4d(self):
        config['world.dim'] = 4
        self.run_experiment(config)


if __name__ == '__main__':
    unittest.main()
