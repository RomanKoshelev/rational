import tensorflow as tf


def flipkernel(kern):
    return kern[(slice(None, None, -1),) * 2 + (slice(None), slice(None))]


def theano_to_tf(tensor):
    # NCHW -> NHWC
    return tf.transpose(tensor, [0, 2, 3, 1])


def tf_to_theano(tensor):
    # NHWC -> NCHW
    return tf.transpose(tensor, [0, 3, 1, 2])
