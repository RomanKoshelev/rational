import numpy as np

from reinforcement_learning import IWorld


class RandomWorld(IWorld):
    def __init__(self):
        self._obs_dim = 3
        self._act_dim = 2

    def step(self, action: np.ndarray) -> (np.ndarray, float, bool):
        assert len(action) == self._act_dim
        return self._random_state(), 0., False

    def reset(self) -> np.ndarray:
        return self._random_state()

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def act_dim(self) -> int:
        return self._act_dim

    def _random_state(self):
        return np.random.random_sample(size=self._obs_dim)
