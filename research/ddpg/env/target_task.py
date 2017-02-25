import numpy as np
from reinforcement_learning import ITask


class TargetTask(ITask):
    def __init__(self, config, bounds):
        self.dim = len(bounds)
        self.bounds = bounds
        self.reward_done = config['task.reward_done']
        self.reward_dist = config['task.reward_dist']
        self.done_dist = config['task.done_dist']

    def init_agent(self, agent: np.ndarray):
        raise NotImplementedError

    def init_target(self, target: np.ndarray):
        raise NotImplementedError

    def make_reward(self, agent: np.ndarray, target: np.ndarray):
        dist = self._target_dist(agent, target)
        return self.reward_dist - dist + (self.reward_done if self.is_done(agent, target) else 0)

    @staticmethod
    def _target_dist(agent: np.ndarray, target: np.ndarray):
        return np.sqrt(np.sum((agent - target) ** 2))

    def is_done(self, agent: np.ndarray, target: np.ndarray):
        return self._target_dist(agent, target) < self.done_dist


class FixedTargetTask(TargetTask):
    def init_agent(self, _):
        return np.zeros((self.dim,))

    def init_target(self, _):
        return self.bounds / 2


class RandomTargetTask(TargetTask):
    def init_agent(self, agent: np.ndarray):
        return agent.copy()

    def init_target(self, _):
        return np.random.random_sample(self.dim) * self.bounds
