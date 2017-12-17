import tensorflow as tf

from caps_layer import CapstoCaps
from caps_layer import Conv2DtoCaps
from config import cfg

num_primary_caps = 32
primary_caps_dim = 8
secondary_caps_dim = 16


class TreeLayer(object):
    def __init__(self, stage, image, labels, prev_layer_caps=None):
        self.image = image
        self.stage = stage
        self.labels = labels
        self.prev_layer_caps = prev_layer_caps

    def __call__(self, n_outputs):
        with tf.variable_scope('tree_layer' + str(self.stage)):
            primary_caps = Conv2DtoCaps(num_primary_caps, primary_caps_dim, self.stage)
            caps_1 = primary_caps(self.image, self.prev_layer_caps)
            secondary_caps = CapstoCaps(n_outputs, secondary_caps_dim, num_primary_caps * cfg.area,
                                        primary_caps_dim, self.stage)
            caps_2 = secondary_caps(caps_1)
            probs = tf.sqrt(tf.reduce_sum(tf.square(caps_2), axis=2, keep_dims=True) + cfg.epsilon)

            layer_loss = margin_loss(probs, self.labels, n_outputs)

            return layer_loss, probs, caps_1


def margin_loss(probs, labels, n_outputs):
    correct_class = tf.square(tf.maximum(0., cfg.m_plus - probs))
    other_classes = cfg.lambd * tf.square(tf.maximum(0., probs - cfg.m_minus))

    correct_class = tf.reshape(correct_class, [cfg.batch_size, n_outputs])
    other_classes = tf.reshape(other_classes, [cfg.batch_size, n_outputs])

    loss = labels * correct_class + (1 - labels) * other_classes
    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
