import numpy as np
import tensorflow as tf

from common.events import EventSystem
from .data import process_gridworld_data

from research.vin_TheAbhiKumar.vin_algorithm.model import VI_Block


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
        print(world_size)
        tf.app.flags.DEFINE_string('input', 'mat_data/gridworld_%d.mat' % world_size, 'Path to data')
        tf.app.flags.DEFINE_integer('imsize', world_size, 'Size of input image')
        # Parameters
        tf.app.flags.DEFINE_float('lr', cfg['train.learning_rate'], 'Learning rate for RMSProp')
        tf.app.flags.DEFINE_integer('epochs', cfg['train.epoches'], 'Maximum epochs to train for')
        tf.app.flags.DEFINE_integer('k', world_size+4, 'Number of value iterations')
        tf.app.flags.DEFINE_integer('ch_i', 2, 'Channels in input layer')
        tf.app.flags.DEFINE_integer('ch_h', 150, 'Channels in initial hidden layer')
        tf.app.flags.DEFINE_integer('ch_q', 10, 'Channels in q layer (~actions)')
        tf.app.flags.DEFINE_integer('batchsize', cfg['train.batch_size'], 'Batch size')
        tf.app.flags.DEFINE_integer('statebatchsize', 10,
                                    'Number of state inputs for each sample (real number, technically is k+1)')
        self.config = tf.app.flags.FLAGS

    def _build(self):
        cfg = self.config

        # symbolic input image tensor where typically first channel is image, second is the reward prior
        self.image = tf.placeholder(tf.float32, name="image", shape=[None, cfg.imsize, cfg.imsize, cfg.ch_i])
        # symbolic input batches of vertical positions
        self.path_ver = tf.placeholder(tf.int32, name="path_ver", shape=[None, cfg.statebatchsize])
        # symbolic input batches of horizontal positions
        self.path_hor = tf.placeholder(tf.int32, name="path_hor", shape=[None, cfg.statebatchsize])
        self.true = tf.placeholder(tf.int32, name="true", shape=[None])

        # Construct model (Value Iteration Network)
        logits, outputs = VI_Block(self.image, self.path_ver, self.path_hor, cfg)

        # Define loss and optimizer
        y = tf.cast(self.true, tf.int64)
        self.err = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y), name='err')
        self.opt = tf.train.RMSPropOptimizer(cfg.lr, epsilon=1e-6, centered=True).minimize(self.err)

        # Test model & calculate accuracy
        pred = tf.cast(tf.argmax(outputs, 1), tf.int32)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(pred, self.true), dtype=tf.float32))

        # gridworld data
        self.gridworld_data = process_gridworld_data(input=cfg.input, imsize=cfg.imsize)

        # Initializing the variables
        self._session.run(tf.global_variables_initializer())

    def train(self):
        cfg = self.config

        batch_size = cfg.batchsize

        x, s1, s2, y, _, _, _, _ = self.gridworld_data

        for epoch in range(int(cfg.epochs)):

            err, acc = 0.0, 0.0
            num_batches = int(x.shape[0] / batch_size)

            for i in range(0, x.shape[0], batch_size):
                j = i + batch_size
                if j <= x.shape[0]:
                    feeder = {
                        self.image: x[i:j],
                        self.path_ver: s1[i:j],
                        self.path_hor: s2[i:j],
                        self.true: y[i * cfg.statebatchsize:j * cfg.statebatchsize]
                    }

                    _, e, a = self._session.run([self.opt, self.err, self.acc], feed_dict=feeder)
                    err += e
                    acc += a

            EventSystem.send('algorithm.train', {
                'epoch': epoch,
                'train_err': err / num_batches,
                'train_acc': acc / num_batches,
            })

    def eval(self):
        _, _, _, _, x, s1, s2, y = self.gridworld_data

        acc = self.acc.eval({
            self.image: x,
            self.path_ver: s1,
            self.path_hor: s2,
            self.true: y
        })

        EventSystem.send('algorithm.eval', {
            'acc': acc,
        })

        return acc
