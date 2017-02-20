import numpy as np


class IWorld(object):
    @property
    def obs_dim(self) -> int:
        raise NotImplementedError

    @property
    def act_dim(self) -> int:
        raise NotImplementedError

    def reset(self) -> np.ndarray:  # state
        raise NotImplementedError

    def step(self, action: np.ndarray) -> (np.ndarray, float, bool):  # state, reward, done
        raise NotImplementedError
