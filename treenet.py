import tensorflow as tf

from config import cfg
from treelayer import TreeLayer

caps_size = 5


class TreeNet(object):

    def __init__(self, X, Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, depth):
        self.X = X
        self.Y_1 = Y_1
        self.Y_2 = Y_2
        self.Y_3 = Y_3
        self.Y_4 = Y_4
        self.Y_5 = Y_5
        self.Y_6 = Y_6
        self.depth = depth
        self.predict()
        self.optimizer = tf.train.AdamOptimizer(cfg.lr)
        self.train_op = self.optimizer.minimize(self.total_loss)

        self.preds_1 = tf.equal(tf.argmax(tf.nn.softmax(tf.squeeze(self.probs_1), dim=1), axis=1),
                                tf.argmax(self.Y_1, axis=1))
        self.accuracy_1 = tf.reduce_mean(tf.cast(self.preds_1, dtype=tf.float32))

        self.preds_2 = tf.equal(tf.argmax(tf.nn.softmax(tf.squeeze(self.probs_2), dim=1), axis=1),
                                tf.argmax(self.Y_2, axis=1))
        self.accuracy_2 = tf.reduce_mean(tf.cast(self.preds_2, dtype=tf.float32))

        self.preds_3 = tf.equal(tf.argmax(tf.nn.softmax(tf.squeeze(self.probs_3), dim=1), axis=1),
                                tf.argmax(self.Y_3, axis=1))
        self.accuracy_3 = tf.reduce_mean(tf.cast(self.preds_3, dtype=tf.float32))

        self.preds_4 = tf.equal(tf.argmax(tf.nn.softmax(tf.squeeze(self.probs_4), dim=1), axis=1),
                                tf.argmax(self.Y_4, axis=1))
        self.accuracy_4 = tf.reduce_mean(tf.cast(self.preds_4, dtype=tf.float32))

        self.preds_5 = tf.equal(tf.argmax(tf.nn.softmax(tf.squeeze(self.probs_5), dim=1), axis=1),
                                tf.argmax(self.Y_5, axis=1))
        self.accuracy_5 = tf.reduce_mean(tf.cast(self.preds_5, dtype=tf.float32))

        self.preds_6 = tf.equal(tf.argmax(tf.nn.softmax(tf.squeeze(self.probs_6), dim=1), axis=1),
                                tf.argmax(self.Y_6, axis=1))
        self.accuracy_6 = tf.reduce_mean(tf.cast(self.preds_6, dtype=tf.float32))

        self.accuracies = [self.accuracy_1, self.accuracy_2, self.accuracy_3,
                           self.accuracy_4, self.accuracy_5, self.accuracy_6]

    def predict(self):
        x = tf.reshape(self.X, shape=[-1, 3, 32, 32])
        x = tf.transpose(x, (0, 2, 3, 1))
        x = tf.layers.conv2d(x, filters=32, kernel_size=7, strides=1, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=128, kernel_size=7, strides=1, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=1, padding='valid', activation=tf.nn.relu)
        print(x)
        branch_1 = TreeLayer(stage=1, image=x, labels=self.Y_1)
        layer_1_loss, self.probs_1, layer_1_caps = branch_1(n_outputs=6)
        if self.depth == 1:
            self.total_loss = layer_1_loss
        else:
            x_2 = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
            x_2_caps = tf.reshape(layer_1_caps, [cfg.batch_size, caps_size, caps_size, -1])
            branch_2 = TreeLayer(stage=2, image=x_2, labels=self.Y_2, prev_layer_caps=x_2_caps)
            layer_2_loss, self.probs_2, layer_2_caps = branch_2(n_outputs=12)
            if self.depth == 2:
                self.total_loss = layer_1_loss + layer_2_loss
            else:
                x_3 = tf.layers.conv2d(x_2, filters=256, kernel_size=3, strides=1, padding='same',
                                       activation=tf.nn.relu)
                x_3_caps = tf.reshape(layer_2_caps, [cfg.batch_size, caps_size, caps_size, -1])
                branch_3 = TreeLayer(stage=3, image=x_3, labels=self.Y_3, prev_layer_caps=x_3_caps)
                layer_3_loss, self.probs_3, layer_3_caps = branch_3(n_outputs=33)
                if self.depth == 3:
                    self.total_loss = layer_1_loss + layer_2_loss + layer_3_loss
                else:
                    x_4 = tf.layers.conv2d(x_3, filters=256, kernel_size=3, strides=1, padding='same',
                                           activation=tf.nn.relu)
                    x_4_caps = tf.reshape(layer_3_caps, [cfg.batch_size, caps_size, caps_size, -1])
                    branch_4 = TreeLayer(stage=4, image=x_4, labels=self.Y_4, prev_layer_caps=x_4_caps)
                    layer_4_loss, self.probs_4, layer_4_caps = branch_4(n_outputs=34)
                    if self.depth == 4:
                        self.total_loss = layer_1_loss + layer_2_loss + layer_3_loss + layer_4_loss
                    else:
                        x_5 = tf.layers.conv2d(x_4, filters=256, kernel_size=3, strides=1, padding='same',
                                               activation=tf.nn.relu)
                        x_5_caps = tf.reshape(layer_4_caps, [cfg.batch_size, caps_size, caps_size, -1])
                        branch_5 = TreeLayer(stage=5, image=x_5, labels=self.Y_5, prev_layer_caps=x_5_caps)
                        layer_5_loss, self.probs_5, layer_5_caps = branch_5(n_outputs=25)
                        if self.depth == 5:
                            self.total_loss = layer_1_loss + layer_2_loss + layer_3_loss + layer_4_loss + layer_5_loss
                        else:
                            x_6 = tf.layers.conv2d(x_5, filters=256, kernel_size=3, strides=1, padding='same',
                                                   activation=tf.nn.relu)
                            x_6_caps = tf.reshape(layer_5_caps, [cfg.batch_size, caps_size, caps_size, -1])
                            branch_6 = TreeLayer(stage=6, image=x_6, labels=self.Y_6, prev_layer_caps=x_6_caps)
                            layer_6_loss, self.probs_6, layer_6_caps = branch_6(n_outputs=12)
                            self.total_loss = layer_1_loss + layer_2_loss + layer_3_loss + layer_4_loss + layer_5_loss + layer_6_loss
