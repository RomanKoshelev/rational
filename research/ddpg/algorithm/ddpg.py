import numpy as np
import tensorflow as tf

from common.events import EventSystem
from reinforcement_learning import IWorld
from .actor import ActorNetwork
from .buffer import ReplayBuffer
from .critic import CriticNetwork
from .tf_helper import TfHelper
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
        self.helper = TfHelper(scope)
        self.expl = OUNoise(self.world.act_dim, 0, self.noise_sigma, self.noise_theta)
        self._create_networks(config, scope)

    def eval(self, episodes, steps):
        state = None
        reward = 0
        done = 0
        for ep in range(episodes):
            state = self.world.reset()
            for __ in range(steps):
                a = self._predict(state)
                state, r, d = self.world.step(a)
                reward += r
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

    def train(self, episodes, steps):
        self._do_train(episodes, steps)

    def _do_train(self, episodes, steps):
        for e in range(episodes):
            state = self.world.reset()
            reward = 0
            done = False
            qmax = []
            for _ in range(steps):
                if not done:
                    state, r, done = self._play(state)
                    reward += r
                q = self._learn()
                qmax.append(np.amax(q))
            self._send_train_event(e, state, reward, done, qmax)

    def _create_networks(self, config, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope('actor'):
                self.actor = ActorNetwork(config, self.world.obs_dim, self.world.act_dim)
            with tf.variable_scope('critic'):
                self.critic = CriticNetwork(config, self.world.obs_dim, self.world.act_dim)
        self.helper.initialize_variables()

    def _play(self, s):
        a = self._predict(s) + self.expl.noise()
        s2, r, done = self.world.step(a)
        self._add_to_buffer(s, a, r, s2, done)
        return s2, r, done

    def _learn(self):
        bs, ba, br, bs2, bd = self._get_batch()
        y = self._make_target(br, bs2, bd)
        q = self._update_critic(y, bs, ba)
        self._update_actor(bs)
        self._update_target_networks()
        return q

    def _predict(self, s):
        return self.actor.predict([s])[0]

    def _make_target(self, r, s, done):
        q = self.critic.target_predict(s, self.actor.target_predict(s))
        y = []
        for i in range(len(s)):
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

    def _add_to_buffer(self, s, a, r, s2, done):
        self.buffer.add(s, a, r, s2, done)

    @staticmethod
    def _send_train_event(e, state, reward, done, qmax):
        EventSystem.send('algorithm.train_episode', {
            'episode': e,
            'reward': reward,
            'qmax': np.mean(qmax),
            'state': state,
            'done': done
        })
