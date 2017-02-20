import numpy as np
import tensorflow as tf

from common.events import Events
from reinforcement_learning import IWorld
from .actor import ActorNetwork
from .buffer import ReplayBuffer
from .critic import CriticNetwork
from .helper import Helper
from .ou_noise import OUNoise


class DdpgAlgorithm(object):
    def __init__(self, config, session, world: IWorld, scope=''):
        self.config = config
        self.world = world
        self.buffer = None
        with tf.variable_scope(scope):
            with tf.variable_scope('actor'):
                self.actor = ActorNetwork(config, session, world.obs_dim, world.act_dim)
            with tf.variable_scope('critic'):
                self.critic = CriticNetwork(config, session, world.obs_dim, world.act_dim)

        self.episode = None
        self.helper = Helper(session, scope)
        self.helper.initialize_variables()

    def predict(self, s):
        return self.actor.predict([s])[0]

    def train(self, episodes, steps):
        first_episode = self.episode + 1 if self.episode is not None else 0
        expl = self._create_exploration()
        done = False

        if self.buffer is None:
            self.buffer = self._create_buffer()

        for self.episode in range(first_episode, episodes + 1):
            s = self.world.reset()

            nrate = self._get_noise_rate(self.episode, episodes)
            reward = 0
            qmax = []

            if self.episode % 100 == 0:
                expl.reset()

            for _ in range(steps):
                # play
                a = self._make_noisy_action(s, expl.noise(), nrate)
                s2, r, done = self.world.step(a)
                self.buffer.add(s, a, r, s2, done)
                s = s2

                # learn
                bs, ba, br, bs2, bd = self._get_batch()
                y = self._make_target(br, bs2, bd)
                q = self._update_critic(y, bs, ba)
                self._update_actor(bs)
                self._update_target_networks()

                # statistic
                reward += r
                qmax.append(q)

                if done:
                    break

            Events.send('algorithm.train_episode_end', {
                'episode': self.episode,
                'reward': reward,
                'nrate': nrate,
                'qmax': np.mean(qmax),
                'done': done
            })

    def _make_noisy_action(self, s, noise, noise_rate) -> np.ndarray:
        def add_noise(a, n, k):
            return (1 - k) * a + k * n
        act = self.actor.predict([s])[0]
        act = add_noise(act, noise, noise_rate)
        return act

    def _make_target(self, r, s2, done):
        q = self.critic.target_predict(s2, self.actor.target_predict(s2))
        y = []
        for i in range(len(s2)):
            if done[i]:
                y.append(r[i])
            else:
                y.append(r[i] + self.config['ddpg.gamma'] * q[i])
        return np.reshape(y, (-1, 1))

    def _update_critic(self, y, s, a):
        q, _ = self.critic.train(y, s, a)
        return np.amax(q)

    def _update_actor(self, s):
        grads = self.critic.gradients(s, self.actor.predict(s))
        self.actor.train(s, grads)

    def _update_target_networks(self):
        self.actor.target_train()
        self.critic.target_train()

    def _get_batch(self):
        batch = self.buffer.get_batch(self.config['ddpg.batch_size'])
        s, a, r, s2, done = zip(*batch)
        return s, a, r, s2, done

    def _create_exploration(self):
        return OUNoise(self.world.act_dim, mu=0,
                       sigma=self.config['ddpg.noise_sigma'],
                       theta=self.config['ddpg.noise_theta'])

    def _get_noise_rate(self, episode, episodes):
        return self.config['ddpg.noise_rate_method'](episode / float(episodes))

    def _create_buffer(self):
        return ReplayBuffer(self.config['ddpg.buffer_size'])
