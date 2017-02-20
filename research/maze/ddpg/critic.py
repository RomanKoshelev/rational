import tensorflow as tf


class CriticNetwork(object):
    def __init__(self, config, sess, state_size, action_size):
        self.h1 = config['ddpg.critic_h1']
        self.h2 = config['ddpg.critic_h2']
        self.l2 = config['ddpg.critic_l2']
        self.lr = config['ddpg.critic_lr']
        self.tau = config['ddpg.critic_tau']

        self.sess = sess

        with tf.variable_scope('master'):
            self.state, self.action, self.out, self.weights = \
                self.create_critic_network(state_size, action_size)

        with tf.variable_scope('target'):
            self.target_state, self.target_action, self.target_update, self.target_net, self.target_out = \
                self.crate_critic_target_network(state_size, action_size, self.weights)

        # TRAINING
        self.y = tf.placeholder(tf.float32, [None, 1], name='y')
        self.error = tf.reduce_mean(tf.square(self.y - self.out))
        self.weight_decay = tf.add_n([self.l2 * tf.nn.l2_loss(var) for var in self.weights])
        self.loss = self.error + self.weight_decay
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # GRADIENTS for policy update
        self.action_grads = tf.gradients(self.out, self.action)

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def train(self, y, states, actions):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.y: y,
            self.state: states,
            self.action: actions
        })

    def predict(self, states, actions):
        return self.sess.run(self.out, feed_dict={
            self.state: states,
            self.action: actions
        })

    def target_predict(self, states, actions):
        return self.sess.run(self.target_out, feed_dict={
            self.target_state: states,
            self.target_action: actions
        })

    def target_train(self):
        self.sess.run(self.target_update)

    def crate_critic_target_network(self, input_dim, action_dim, net):
        state = tf.placeholder(tf.float32, shape=[None, input_dim], name='state')
        action = tf.placeholder(tf.float32, shape=[None, action_dim], name='action')

        ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        h1 = tf.nn.relu(tf.matmul(state, target_net[0]) + target_net[1])
        h2 = tf.nn.relu(tf.matmul(
            h1, target_net[2]) + tf.matmul(action, target_net[3]) + target_net[4])
        out = tf.identity(tf.matmul(h2, target_net[5]) + target_net[6])

        return state, action, target_update, target_net, out

    def create_critic_network(self, state_dim, action_dim):
        state = tf.placeholder(tf.float32, shape=[None, state_dim], name='state')
        action = tf.placeholder(tf.float32, shape=[None, action_dim], name='action')

        w1 = self.weight_variable([state_dim, self.h1])
        b1 = self.bias_variable([self.h1])
        w2 = self.weight_variable([self.h1, self.h2])
        b2 = self.bias_variable([self.h2])
        w2_action = self.weight_variable([action_dim, self.h2])
        w3 = self.weight_variable([self.h2, 1])
        b3 = self.bias_variable([1])

        h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
        h2 = tf.nn.relu(tf.matmul(h1, w2) + tf.matmul(action, w2_action) + b2)
        out = tf.identity(tf.matmul(h2, w3) + b3)

        return state, action, out, [w1, b1, w2, w2_action, b2, w3, b3]

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.001)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.001, shape=shape)
        return tf.Variable(initial)
