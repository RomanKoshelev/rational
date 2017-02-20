import tensorflow as tf
import pickle
import os


class Helper(object):
    def __init__(self, sess, scope):
        self.sess = sess
        self.scope = scope

    def initialize_variables(self):
        self.sess.run(tf.variables_initializer(self._variables))

    @property
    def _variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    def save_weights(self, path):
        saver = tf.train.Saver(self._variables)
        saver.save(self.sess, path)

    def restore_weights(self, path):
        saver = tf.train.Saver(self._variables)
        if not os.path.exists(path):
            raise ValueError("File not found: '%s'" % path)
        saver.restore(self.sess, path)

    @staticmethod
    def save_state(data_list, path):
        with open(path, 'wb') as f:
            pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def restore_state(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _weights_path(path):
        return os.path.join(path, 'network_weights.ckpt')

    @staticmethod
    def _state_path(path):
        return os.path.join(path, 'algorithm_state.pickle')
