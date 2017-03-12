import numpy as np
import tensorflow as tf

from common.events import EventSystem
from .data import process_gridworld_data

from research.vin_TheAbhiKumar.vin_algorithm.model import VI_Untied_Block, VI_Block


class VinAlgorithm(object):
    def __init__(self, cfg):
        np.random.seed(0)
        self.config = None
        self._session = tf.get_default_session()
        self._config(cfg)
        self._build()

    def _config(self, cfg):
        # Data
        world_size = cfg['world.size']
        tf.app.flags.DEFINE_string('input', 'mat_data/gridworld_%d.mat' % world_size, 'Path to data')
        tf.app.flags.DEFINE_integer('imsize', world_size, 'Size of input image')
        # Parameters
        tf.app.flags.DEFINE_float('lr', 0.001, 'Learning rate for RMSProp')
        tf.app.flags.DEFINE_integer('epochs', cfg['train.epoches'], 'Maximum epochs to train for')
        tf.app.flags.DEFINE_integer('k', 10, 'Number of value iterations')
        tf.app.flags.DEFINE_integer('ch_i', 2, 'Channels in input layer')
        tf.app.flags.DEFINE_integer('ch_h', 150, 'Channels in initial hidden layer')
        tf.app.flags.DEFINE_integer('ch_q', 10, 'Channels in q layer (~actions)')
        tf.app.flags.DEFINE_integer('batchsize', 12, 'Batch size')
        tf.app.flags.DEFINE_integer('statebatchsize', 10,
                                    'Number of state inputs for each sample (real number, technically is k+1)')
        tf.app.flags.DEFINE_boolean('untied_weights', False, 'Untie weights of VI network')
        self.config = tf.app.flags.FLAGS

    def _build(self):
        cfg = self.config

        # symbolic input image tensor where typically first channel is image, second is the reward prior
        self.x_pl = tf.placeholder(tf.float32, name="x", shape=[None, cfg.imsize, cfg.imsize, cfg.ch_i])
        # symbolic input batches of vertical positions
        self.s1_pl = tf.placeholder(tf.int32, name="s1", shape=[None, cfg.statebatchsize])
        # symbolic input batches of horizontal positions
        self.s2_pl = tf.placeholder(tf.int32, name="s2", shape=[None, cfg.statebatchsize])
        self.y_pl = tf.placeholder(tf.int32, name="y", shape=[None])

        # Construct model (Value Iteration Network)
        if cfg.untied_weights:
            logits, nn = VI_Untied_Block(self.x_pl, self.s1_pl, self.s2_pl, cfg)
        else:
            logits, nn = VI_Block(self.x_pl, self.s1_pl, self.s2_pl, cfg)

        # Define loss and optimizer
        y_ = tf.cast(self.y_pl, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_, name='cross_entropy')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)
        self.cost = tf.add_n(tf.get_collection('losses'), name='total_loss')
        # self.optimizer = tf.train.RMSPropOptimizer(cfg.lr, epsilon=1e-6, centered=True).minimize(self.cost)
        self.optimizer = tf.train.AdamOptimizer(cfg.lr).minimize(self.cost)

        # Test model & calculate accuracy
        cp = tf.cast(tf.argmax(nn, 1), tf.int32)
        self.err = tf.reduce_mean(tf.cast(tf.not_equal(cp, self.y_pl), dtype=tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(cp, self.y_pl), dtype=tf.float32))

        # gridworld data
        self.gridworld_data = process_gridworld_data(input=cfg.input, imsize=cfg.imsize)

        # Initializing the variables
        self._session.run(tf.global_variables_initializer())

    def train(self):
        cfg = self.config

        batch_size = cfg.batchsize

        x, s1, s2, y, _, _, _, _ = self.gridworld_data

        for epoch in range(int(cfg.epochs)):

            err, cost = 0.0, 0.0
            num_batches = int(x.shape[0] / batch_size)

            for i in range(0, x.shape[0], batch_size):
                j = i + batch_size
                if j <= x.shape[0]:
                    feeder = {
                        self.x_pl: x[i:j],
                        self.s1_pl: s1[i:j],
                        self.s2_pl: s2[i:j],
                        self.y_pl: y[i * cfg.statebatchsize:j * cfg.statebatchsize]
                    }

                    _, c, e = self._session.run([self.optimizer, self.cost, self.err], feed_dict=feeder)
                    err += e
                    cost += c

            EventSystem.send('algorithm.train', {
                'epoch': epoch,
                'train_cost': cost / num_batches,
                'train_error': err / num_batches,
            })

    def eval(self):
        _, _, _, _, x, s1, s2, y = self.gridworld_data

        acc = self.accuracy.eval({
            self.x_pl: x,
            self.s1_pl: s1,
            self.s2_pl: s2,
            self.y_pl: y
        })

        EventSystem.send('algorithm.eval', {
            'accuracy': acc,
        })

        return acc
