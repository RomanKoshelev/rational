import tensorflow as tf


class ActorNetwork(object):
    def __init__(self, config, sess, state_size, action_size):
        self.h1 = config['alg.actor.h1']
        self.h2 = config['alg.actor.h2']
        self.l2 = config['alg.actor.l2']
        self.lr = config['alg.actor.lr']
        self.tau = config['alg.actor.tau']

        self.sess = sess

        with tf.variable_scope('master'):
            self.state, self.out, self.weights = self.create_actor_network(state_size, action_size)

        with tf.variable_scope('target'):
            self.target_state, self.target_update, self.target_net, self.target_out = \
                self.crate_actor_target_network(state_size, self.weights)

        # TRAINING
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size], name='action_gradient')
        self.grads = tf.gradients(self.out, self.weights, -self.action_gradient)
        grads = zip(self.grads, self.weights)
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def predict(self, states):
        return self.sess.run(self.out, feed_dict={
            self.state: states
        })

    def target_predict(self, states):
        return self.sess.run(self.target_out, feed_dict={
            self.target_state: states
        })

    def target_train(self):
        self.sess.run(self.target_update)

    def crate_actor_target_network(self, input_dim, weights):
        state = tf.placeholder(tf.float32, shape=[None, input_dim], name='state')

        ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)
        target_update = ema.apply(weights)
        target_weights = [ema.average(x) for x in weights]

        h1 = tf.nn.relu(tf.matmul(state, target_weights[0]) + target_weights[1])
        h2 = tf.nn.relu(tf.matmul(h1, target_weights[2]) + target_weights[3])
        out = tf.tanh(tf.matmul(h2, target_weights[4]) + target_weights[5])

        return state, target_update, target_weights, out

    def create_actor_network(self, input_dim, output_dim):
        state = tf.placeholder(tf.float32, shape=[None, input_dim], name='state')

        w1 = self.weight_variable([input_dim, self.h1])
        b1 = self.bias_variable([self.h1])
        w2 = self.weight_variable([self.h1, self.h2])
        b2 = self.bias_variable([self.h2])
        w3 = self.weight_variable([self.h2, output_dim])
        b3 = self.bias_variable([output_dim])

        h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
        out = tf.tanh(tf.matmul(h2, w3) + b3)

        return state, out, [w1, b1, w2, b2, w3, b3]

    @staticmethod
    def weight_variable(shape):
        initial = tf.random_uniform(shape, minval=-0.05, maxval=0.05)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)
