import numpy as np

from common.events import Events
from reinforcement_learning import IWorld


class TargetWorld(IWorld):
    def __init__(self):
        self.limit = np.array([60., 40.])
        self.target = np.array([10., 10.])
        self.agent = np.array([0., 0.])
        self._obs_dim = len(self._get_state())
        self._act_dim = 2

    def step(self, a: np.ndarray) -> (np.ndarray, float, bool):
        self._do_action(a)
        return (self._get_state(),
                self._make_reward(),
                self._is_done())

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

    def _is_done(self):
        return self.agent[0] == self.target[0] and self.agent[1] == self.target[1]

    def _make_reward(self):
        dist = np.sqrt(np.sum((self.agent - self.target) ** 2))
        return 1/max(dist, .01)

    def _do_action(self, a):
        a = np.minimum(a, np.array([+1., +1.]))
        a = np.maximum(a, np.array([-1., -1.]))
        self.agent += a
        self.agent = np.minimum(self.agent, self.limit)
        self.agent = np.maximum(self.agent, np.array([0., 0.]))
        Events.send('world.action', {
            'agent': self.agent,
            'action': a
        })
