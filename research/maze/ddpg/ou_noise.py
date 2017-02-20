# --------------------------------------
# Ornstein-Uhlenbeck Noise
# Author: Flood Sung
# Date: 2016.5.4
# Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
# --------------------------------------

import numpy as np
import numpy.random as nr


# noinspection PyArgumentList
class OUNoise:

    def __init__(self, action_dimension, mu=0., sigma=.3, theta=0.15):
        self.action_dimension = action_dimension
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state  # type: np.ndarray
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state


if __name__ == '__main__':
    ou = OUNoise(3, mu=0., sigma=.1, theta=.01)
    states = []
    for i in range(100000):
        states.append(ou.noise())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
