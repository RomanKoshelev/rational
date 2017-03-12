import unittest

from common.timers.event_timer import EventTimer
from research.vin_TheAbhiKumar.config import config
from research.vin_TheAbhiKumar.utils.logger import VinTrainLogger
from research.vin_TheAbhiKumar.vin_algorithm.train import train


class TestVin(unittest.TestCase):

    # noinspection PyMethodMayBeStatic
    def run_experiment(self, cfg):
        with EventTimer('algorithm.train_epoch'), VinTrainLogger():
            train()

    def test_8x8(self):
        self.run_experiment(config)


if __name__ == '__main__':
    unittest.main()
