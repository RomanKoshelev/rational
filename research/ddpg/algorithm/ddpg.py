import numpy as np
import tensorflow as tf

from common.events import EventSystem
from reinforcement_learning import IWorld
from .actor import ActorNetwork
from .buffer import ReplayBuffer
from .critic import CriticNetwork
from .store_helper import StoreHelper
from .ou_noise import OUNoise


class Ddpg(object):
    def __init__(self, config, world: IWorld, scope=''):
        self.noise_theta = config['ddpg.noise_theta']
        self.noise_sigma = config['ddpg.noise_sigma']
        self.buffer_size = config['ddpg.buffer_size']
        self.batch_size = config['ddpg.batch_size']

        self.gamma = config['ddpg.gamma']
        self.world = world
        self.buffer = ReplayBuffer(self.buffer_size)
        self.helper = StoreHelper(scope)
        self.expl = OUNoise(self.world.act_dim, 0, self.noise_sigma, self.noise_theta)

        with tf.variable_scope(scope):
            with tf.variable_scope('actor'):
                self.actor = ActorNetwork(config, world.obs_dim, world.act_dim)
            with tf.variable_scope('critic'):
                self.critic = CriticNetwork(config, world.obs_dim, world.act_dim)

        self.helper.initialize_variables()

    def predict(self, s):
        return self.actor.predict([s])[0]

    def train(self, episodes, steps):
        for e in range(episodes):
            state = s = self.world.reset()
            reward = 0
            done = False
            qmax = []

            for _ in range(steps):
                if not done:
                    a = self.predict(s)
                    a = self._add_noise(a, 1.)
                    s2, r, done = self.world.step(a)
                    self.buffer.add(s, a, r, s2, done)
                    state = s = s2
                    reward += r

                bs, ba, br, bs2, bd = self._get_batch()
                y = self._make_target(br, bs2, bd)
                q = self._update_critic(y, bs, ba)
                self._update_actor(bs)
                self._update_target_networks()
                qmax.append(np.amax(q))

            EventSystem.send('algorithm.train_episode', {
                'episode': e,
                'reward': reward,
                'qmax': np.mean(qmax),
                'state': state,
                'done': done
            })

    def eval(self, episodes, steps):
        done = 0
        reward = 0
        state = None
        for _ in range(episodes):
            s = self.world.reset()
            for __ in range(steps):
                a = self.actor.predict([s])[0]
                s, r, d = self.world.step(a)
                reward += r
                state = s
                if d:
                    done += 1
                    break

        reward /= float(episodes)
        done /= float(episodes)
        EventSystem.send('algorithm.eval', {
            'ave_reward': reward,
            'ave_done': done,
            'state': state,
        })
        return reward, done

    def _add_noise(self, a, nr) -> np.ndarray:
        n = self.expl.noise()
        # return (1 - nr) * a + nr * n
        return a + nr * n

    def _make_target(self, r, s2, done):
        q = self.critic.target_predict(s2, self.actor.target_predict(s2))
        y = []
        for i in range(len(s2)):
            if done[i]:
                y.append(r[i])
            else:
                y.append(r[i] + self.gamma * q[i])
        return np.reshape(y, (-1, 1))

    def _update_critic(self, y, s, a):
        q, _ = self.critic.train(y, s, a)
        return q

    def _update_actor(self, s):
        grads = self.critic.gradients(s, self.actor.predict(s))
        self.actor.train(s, grads)

    def _update_target_networks(self):
        self.actor.target_train()
        self.critic.target_train()

    def _get_batch(self):
        batch = self.buffer.get_batch(self.batch_size)
        s, a, r, s2, done = zip(*batch)
        return s, a, r, s2, done
