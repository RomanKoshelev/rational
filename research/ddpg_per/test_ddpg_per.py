import unittest

import tensorflow as tf

from common.events import EventSystem
from common.text_utils import fields
from research.ddpg_per.config import config
from research.ddpg_per.algorithm.ddpg_per import DdpgPer
from research.ddpg_per.utils.logger import TrainLogger
from research.ddpg_per.utils.timer import Timer


class TestDddpgPer(unittest.TestCase):
    def run_experiment(self, cfg):
        with Timer(), TrainLogger(), tf.Session():
            episodes, steps = cfg['train.episodes'], cfg['train.steps']
            EventSystem.subscribe('algorithm.train_episode', lambda _: alg.eval(10, steps))
            alg = DdpgPer(cfg, config['world.class'](config))
            alg.train(episodes, steps)

            r, d = alg.eval(100, steps)
            EventSystem.send('train.summary', ["\n", "-" * 32, fields([
                ['Reward', "%.2f" % r],
                ['Done', "%.0f%%" % (d*100)]
            ], -6)])

            self.assertGreater(r, 200)
            self.assertGreater(d, .10)

    def test_default_config(self):
        self.run_experiment(config)

    def test_world_2d(self):
        config['ddpg.buffer_size'] = 10000
        config['train.episodes'] = 500
        config['world.dim'] = 2
        self.run_experiment(config)


if __name__ == '__main__':
    unittest.main()
