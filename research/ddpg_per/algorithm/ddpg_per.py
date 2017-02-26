import numpy as np
import tensorflow as tf

from common.events import EventSystem
from reinforcement_learning import IWorld
from .experience_memory import ExperienceMemory
from .actor import ActorNetwork
from .critic import CriticNetwork
from .store_helper import StoreHelper
from .ou_noise import OUNoise


class DdpgPer(object):
    def __init__(self, config, world: IWorld, scope=''):
        self.noise_theta = config['ddpg.noise_theta']
        self.noise_sigma = config['ddpg.noise_sigma']
        self.batch_size = config['ddpg.batch_size']
        self.noise_rate_method = config['ddpg.noise_rate_method']

        self.buffer_size = config['ddpg.buffer_size']

        self.gamma = config['ddpg.gamma']
        self.world = world
        self.buffer = ExperienceMemory(self.buffer_size)
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
        done = False
        state = None

        self._ini_buffer()

        for ep in range(episodes):
            s = self.world.reset()

            nrate = self.noise_rate_method(ep / float(episodes))
            reward = 0
            qmax = []

            for _ in range(steps):
                # play
                a = self.predict(s) + self.expl.noise()
                s2, r, done = self.world.step(a)
                s = s2

                # learn
                idx, bs, ba, br, bs2, bd = self._get_batch()
                y, qold = self._make_target(br, bs2, bd)
                q = self._update_critic(y, bs, ba)
                self._update_buffer(idx, q, qold)
                self._update_actor(bs)
                self._update_target_networks()

                # statistic
                reward += r
                qmax.append(np.amax(q))
                state = s

                if done:
                    break

            EventSystem.send('algorithm.train_episode', {
                'episode': ep,
                'reward': reward,
                'nrate': nrate,
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

    def _make_target(self, r, s, done):
        q = self.critic.target_predict(s, self.actor.target_predict(s))
        y = []
        for i in range(len(s)):
            if done[i]:
                y.append(r[i])
            else:
                y.append(r[i] + self.gamma * q[i])
        return np.reshape(y, (-1, 1)), np.reshape(q, (-1, 1))

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
        res = self.buffer.get_batch(self.batch_size)
        (idx, samples) = zip(*res)
        (s, a, r, s2, done) = zip(*samples)
        return idx, s, a, r, s2, done

    def _update_buffer(self, idx, q1, q2):
        assert np.shape(q1) == np.shape(q2)
        err = np.abs(q1 - q2)
        for i in range(len(err)):
            self.buffer.update(idx[i], err[i])

    def _ini_buffer(self):
        s = self.world.reset()
        for i in range(self.buffer_size):
            if self.buffer_size > 10000 and i % 1000 == 0:
                print(i)
            a = self.predict(s) + self.expl.noise()
            s2, r, done = self.world.step(a)
            error = np.abs(r)
            sample = (s, a, r, s2, done)
            self.buffer.add(error, sample)
            s = s2
