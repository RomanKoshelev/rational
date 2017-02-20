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
        alg = DdpgAlgorithm(config, tf.Session(), RandomWorld())
        alg.train(10, 100)
        self.assertTrue(True)

    @staticmethod
    def run_experiment(cfg):
        Logger()
        r, q = None, None

        def store_rq(data):
            nonlocal r, q
            r = data['reward']
            q = data['qmax']

        Events.subscribe('algorithm.train_episode', store_rq)
        alg = DdpgAlgorithm(cfg, tf.Session(), TargetWorld(config))
        alg.train(cfg['train.episodes'], cfg['train.steps'])
        return r, q

    def test_default_config(self):
        config['train.episodes'] = 2000
        config['train.steps'] = 10
        config['ddpg.buffer_size'] = 10 * 1000
        config['ddpg.actor_lr'] = 1e-4
        config['ddpg.critic_lr'] = 1e-3
        config['ddpg.actor_tau'] = 1e-3
        config['ddpg.critic_tau'] = 1e-3
        config['ddpg.noise_sigma'] = .5
        config['ddpg.noise_theta'] = 0.15
        r, q = self.run_experiment(config)
        self.assertGreater(r, 90)
        self.assertGreater(q, 900)

    def test_buffer_size(self):
        config['train.buffer_size'] = 2*1000
        r, q = self.run_experiment(config)
        self.assertGreater(r, 90)
        self.assertGreater(q, 800)


if __name__ == '__main__':
    unittest.main()
