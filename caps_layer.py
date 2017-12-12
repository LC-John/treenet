import numpy as np
import tensorflow as tf

from config import cfg


class CapstoCaps(object):

    def __init__(self, n_outputs, output_dim, n_inputs, input_dim, stage):
        self.n_outputs = n_outputs
        self.output_dim = output_dim
        self.n_inputs = n_inputs
        self.input_dim = input_dim
        self.stage = stage

    def __call__(self, input_caps):
        with tf.variable_scope('caps_to_caps' + str(self.stage)):
            input_caps = tf.tile(input_caps, [1, 1, self.n_outputs, 1, 1])
            W_caps = tf.get_variable('W_caps',
                                     shape=[1, self.n_inputs, self.n_outputs, self.input_dim, self.output_dim],
                                     dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=cfg.stddev))
            W_caps_tiled = tf.tile(W_caps, [cfg.batch_size, 1, 1, 1, 1])
            output_caps = tf.matmul(W_caps_tiled, input_caps, transpose_a=True)
            output_caps_norm, full_caps = routing(output_caps, cfg.r_iters, self.n_inputs, self.n_outputs)
            return tf.squeeze(output_caps_norm), full_caps


class Conv2DtoCaps(object):
    def __init__(self, n_outputs, output_dim, stage):
        self.n_outputs = n_outputs
        self.output_dim = output_dim
        self.stage = stage

    def __call__(self, x, prev_layer_caps=None):
        with tf.variable_scope('conv_to_caps' + str(self.stage)):
            x = tf.layers.conv2d(inputs=x, filters=self.n_outputs * self.output_dim,
                                 kernel_size=7, strides=2, padding='valid', activation=tf.nn.relu)
            if self.stage > 1:
                x = tf.concat([x, prev_layer_caps], axis=3)
                x = tf.layers.conv2d(x, filters=self.n_outputs * self.output_dim, kernel_size=1, strides=1,
                                     padding='valid', activation=tf.nn.relu)
            caps = tf.reshape(x, (cfg.batch_size, self.n_outputs * cfg.area, 1, self.output_dim, 1))
            caps_norm = squash(caps, squash_dim=3)
            return caps_norm


def squash(x, squash_dim):
    # x_norm = tf.norm(x, axis=squash_dim, keep_dims=True)
    x_norm_squared = tf.reduce_sum(tf.square(x), axis=squash_dim, keep_dims=True)
    v = x_norm_squared / (1 + x_norm_squared) / tf.sqrt(x_norm_squared + cfg.epsilon)
    return x * v


def routing(x, n_iters, n_inputs, n_outputs):
    b = tf.constant(np.zeros((cfg.batch_size, n_inputs, n_outputs, 1, 1), dtype=np.float32))
    x_temp = tf.stop_gradient(x, name='x_temp')
    for i in range(n_iters):
        with tf.variable_scope('routing' + str(i)):
            c = tf.nn.softmax(b, dim=2)
            with tf.variable_scope('iter' + str(i)):
                if i < n_iters - 1:
                    s = tf.multiply(c, x_temp)
                    s = tf.reduce_sum(s, axis=1, keep_dims=True)
                    v = squash(s, 3)
                    v_tiled = tf.tile(v, [1, n_inputs, 1, 1, 1])
                    b += tf.matmul(x_temp, v_tiled, transpose_a=True)
                elif i == n_iters - 1:
                    s_1 = tf.multiply(c, x)
                    s_2 = tf.reduce_sum(s_1, axis=1, keep_dims=True)
                    v = squash(s_2, 3)
                    return v, s_1
