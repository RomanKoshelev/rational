import numpy as np

from reinforcement_learning import IWorld
from research.ddpg.algorithm.ddpg import Ddpg
from .experience_memory import ExperienceMemory


class DdpgPer(Ddpg):
    def __init__(self, config, world: IWorld, scope=''):
        super().__init__(config, world, scope)
        self.buffer = ExperienceMemory(self.buffer_size, config['per.degree'])

    def train(self, episodes, steps):
        self._init_buffer()
        self._do_train(episodes, steps)

    def _learn(self):
        idx, bs, ba, br, bs2, bd = self._get_batch()
        y, qold = self._make_target(br, bs2, bd)
        q = self._update_critic(y, bs, ba)
        self._update_actor(bs)
        self._update_target_networks()
        self._update_buffer(idx, q, qold)
        return q

    def _make_target(self, r, s, done):
        q = self.critic.target_predict(s, self.actor.target_predict(s))
        y = []
        for i in range(len(s)):
            if done[i]:
                y.append(r[i])
            else:
                y.append(r[i] + self.gamma * q[i])

        return np.reshape(y, (-1, 1)), np.reshape(q, (-1, 1))

    def _init_buffer(self):
        if self.buffer_size > 10000:
            print('Init buffer...')
        s = self.world.reset()
        for i in range(self.buffer_size):
            a = self.expl.noise()
            s2, r, done = self.world.step(a)
            self._add_to_buffer(s, a, r, s2, done)
            s = s2

    def _add_to_buffer(self, s, a, r, s2, done):
        self.buffer.add(np.abs(r), (s, a, r, s2, done))

    def _update_buffer(self, idx, q1, q2):
        assert np.shape(q1) == np.shape(q2)
        err = np.abs(q1 - q2)
        for i in range(len(err)):
            self.buffer.update(idx[i], err[i])

    def _get_batch(self):
        res = self.buffer.get_batch(self.batch_size)
        (idx, samples) = zip(*res)
        (s, a, r, s2, done) = zip(*samples)
        return idx, s, a, r, s2, done
