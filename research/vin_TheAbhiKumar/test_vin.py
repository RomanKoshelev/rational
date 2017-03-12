import unittest
import tensorflow as tf

from common.events import EventSystem
from common.text_utils import fields
from common.timers.event_timer import EventTimer
from research.vin_TheAbhiKumar.config import config
from research.vin_TheAbhiKumar.utils.logger import VinTrainLogger
from research.vin_TheAbhiKumar.vin_algorithm.vin_algorithm import VinAlgorithm


class TestVin(unittest.TestCase):

    # noinspection PyMethodMayBeStatic
    def run_experiment(self, cfg):
        with EventTimer('algorithm.train'), VinTrainLogger(), tf.Session():
            EventSystem.subscribe('algorithm.train', lambda _: alg.eval())
            alg = VinAlgorithm(cfg)
            alg.train()

            acc = alg.eval()
            EventSystem.send('train.summary', ["\n", "-" * 32, fields([
                ['Accuracy', "%.2f%%" % (acc*100)],
            ], -6)])
            self.assertGreater(acc, .9)

    def test_8x8(self):
        self.run_experiment(config)

    def test_16x16(self):
        config['world_size'] = 16
        config['train.epoches'] = 100
        self.run_experiment(config)

    def test_28x28(self):
        config['world_size'] = 28
        config['train.epoches'] = 200
        self.run_experiment(config)


if __name__ == '__main__':
    unittest.main()
