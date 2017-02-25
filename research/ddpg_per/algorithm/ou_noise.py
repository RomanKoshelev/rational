import numpy as np


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

    # noinspection PyArgumentList
    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


if __name__ == '__main__':
    ou = OUNoise(3, mu=0., sigma=.5, theta=.15)
    states = []
    for i in range(1000):
        states.append(ou.noise())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
