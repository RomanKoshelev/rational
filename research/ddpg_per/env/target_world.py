import numpy as np

from reinforcement_learning import IWorld
from .target_task import TargetTask


class TargetWorld(IWorld):
    def __init__(self, config):
        self.dim = config['world.dim']
        self.agent_step = config['world.agent_step']
        self.bounds = np.full((self.dim,), config['world.size'])

        self.target = np.zeros((self.dim,))
        self.agent = np.zeros((self.dim,))

        self.task = self._make_task(config)

        self._obs_dim = len(self._get_state())
        self._act_dim = self.dim

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def act_dim(self) -> int:
        return self._act_dim

    def step(self, a: np.ndarray) -> (np.ndarray, float, bool):
        self._do_action(a)
        state = self._get_state()
        return (state,
                self.task.make_reward(self.agent, self.target),
                self.task.is_done(self.agent, self.target))

    def reset(self) -> np.ndarray:
        self.agent = self.task.init_agent(self.agent)
        self.target = self.task.init_target(self.agent)
        return self._get_state()

    def _get_state(self):
        return self.target - self.agent

    def _do_action(self, a):
        a = np.minimum(a, np.full((self.dim,), +self.agent_step))
        a = np.maximum(a, np.full((self.dim,), -self.agent_step))
        self.agent += a
        self.agent = np.minimum(self.agent, self.bounds)
        self.agent = np.maximum(self.agent, np.zeros((self.dim,)))

    def _make_task(self, config) -> TargetTask:
        return config['task.class'](config, self.bounds)
