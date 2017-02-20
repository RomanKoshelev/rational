import numpy as np

from common.events import Events
from reinforcement_learning import IWorld


class TargetWorld(IWorld):
    def __init__(self, config):
        self.done_dist = config['world.done_dist']
        self.limit = np.array([20., 20.])
        self.target = np.array([10., 10.])
        self.agent = np.array([0., 0.])
        self._obs_dim = len(self._get_state())
        self._act_dim = 2

    def step(self, a: np.ndarray) -> (np.ndarray, float, bool):
        self._do_action(a)
        return (self._get_state(),
                self._make_reward(),
                self._is_done)

    def reset(self) -> np.ndarray:
        return self._get_state()

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def act_dim(self) -> int:
        return self._act_dim

    def _get_state(self):
        return np.append(self.agent, self.target)

    @property
    def _is_done(self):
        return self._target_dist() < .1

    def _make_reward(self):
        dist = self._target_dist()
        return 10-dist + (1000 if self._is_done else 0)

    def _target_dist(self):
        return np.sqrt(np.sum((self.agent - self.target) ** 2)) + 1e-10

    def _do_action(self, a):
        d = self.done_dist
        a = np.minimum(a, np.array([+d, +d]))
        a = np.maximum(a, np.array([-d, -d]))
        self.agent += a
        self.agent = np.minimum(self.agent, self.limit)
        self.agent = np.maximum(self.agent, np.array([0., 0.]))
        Events.send('world.action', {
            'agent': self.agent,
            'action': a
        })
