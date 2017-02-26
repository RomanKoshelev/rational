import unittest

import tensorflow as tf

from common.events import EventSystem
from common.text_utils import fields
from research.ddpg.config import config
from research.ddpg.utils.logger import TrainLogger
from research.ddpg.utils.timer import Timer


class TestDdpg(unittest.TestCase):
    def run_experiment(self, cfg):
        with Timer(), TrainLogger(), tf.Session():
            episodes, steps = cfg['train.episodes'], cfg['train.steps']
            EventSystem.subscribe('algorithm.train_episode', lambda _: alg.eval(10, steps))
            world = cfg['world.class'](cfg)
            alg = cfg['algorithm.class'](cfg, world)
            alg.train(episodes, steps)

            r, d = alg.eval(1000, steps)
            EventSystem.send('train.summary', ["\n", "-" * 32, fields([
                ['Reward', "%.2f" % r],
                ['Done', "%.0f%%" % (d*100)]
            ], -6)])

            self.assertGreater(r, 100)
            self.assertGreater(d, .10)

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

    def test_world_5d(self):
        config['world.dim'] = 5
        self.run_experiment(config)


if __name__ == '__main__':
    unittest.main()
