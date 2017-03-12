import unittest
import tensorflow as tf

from common.timers.event_timer import EventTimer
from research.vin_TheAbhiKumar.config import config
from research.vin_TheAbhiKumar.utils.logger import VinTrainLogger
from research.vin_TheAbhiKumar.vin_algorithm.vin_algorithm import VinAlgorithm


class TestVin(unittest.TestCase):

    # noinspection PyMethodMayBeStatic
    def run_experiment(self, cfg):
        with EventTimer('algorithm.train_epoch'), VinTrainLogger(), tf.Session():
            alg = VinAlgorithm()
            alg.train()

    def test_8x8(self):
        self.run_experiment(config)


if __name__ == '__main__':
    unittest.main()
