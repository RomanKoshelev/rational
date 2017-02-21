import numpy as np

from reinforcement_learning import IWorld


class TargetWorld(IWorld):
    def __init__(self, config):
        self.dim = config['world.dim']
        self.agent_step = config['world.agent_step']
        self.limit = np.full((self.dim,), config['world.size'])
        self.target = np.full((self.dim,), config['world.size']/2)
        self.agent = np.zeros((self.dim,))

        self.reward_done = config['task.reward_done']
        self.reward_dist = config['task.reward_dist']
        self.done_dist = config['task.done_dist']

        self._obs_dim = len(self._get_state())
        self._act_dim = self.dim

    def step(self, a: np.ndarray) -> (np.ndarray, float, bool):
        self._do_action(a)
        return (self._get_state(),
                self._make_reward(),
                self._is_done)

    def reset(self) -> np.ndarray:
        self.agent = np.zeros((self.dim,))
        return self._get_state()

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def act_dim(self) -> int:
        return self._act_dim

    def _get_state(self):
        return self.target - self.agent

    @property
    def _is_done(self):
        return self._target_dist() < self.done_dist

    def _make_reward(self):
        dist = self._target_dist()
        return self.reward_dist-dist + (self.reward_done if self._is_done else 0)

    def _target_dist(self):
        return np.sqrt(np.sum((self.agent - self.target) ** 2))

    def _do_action(self, a):
        a = np.minimum(a, np.full((self.dim,), +self.agent_step))
        a = np.maximum(a, np.full((self.dim,), -self.agent_step))
        self.agent += a
        self.agent = np.minimum(self.agent, self.limit)
        self.agent = np.maximum(self.agent, np.zeros((self.dim,)))
