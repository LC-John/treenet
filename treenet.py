import tensorflow as tf

from config import cfg
from treelayer import TreeLayer

CAPS_SIZE = 5


class TreeNet(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.depth = len(Y)

        self.losses = []
        self.probabilities = []
        self.predictions = []
        self.accuracies = []

        self.predict()

        self.optimizer = tf.train.AdamOptimizer(cfg.lr)
        self.train_ops = [self.optimizer.minimize(loss) for loss in self.losses]

        for i, probability in enumerate(self.probabilities):
            prediction = tf.equal(
                tf.argmax(tf.nn.softmax(tf.squeeze(probability), dim=1), axis=1),
                tf.argmax(self.Y[i], axis=1))
            accuracy = tf.reduce_mean(tf.cast(prediction, dtype=tf.float32))
            self.predictions.append(prediction)
            self.accuracies.append(accuracy)

        for i in range(self.depth):
            prediction = tf.equal(
                tf.argmax(tf.nn.softmax(tf.squeeze(self.probabilities[i]), dim=1), axis=1),
                tf.argmax(self.Y[i], axis=1))
            accuracy = tf.reduce_mean(tf.cast(prediction, dtype=tf.float32))
            self.predictions.append(prediction)
            self.accuracies.append(accuracy)

    def predict(self):
        x = tf.reshape(self.X, shape=[-1, 3, 32, 32])
        x = tf.transpose(x, (0, 2, 3, 1))

        x = tf.layers.conv2d(x, 32, 7, strides=1, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, 3, strides=1, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 128, 7, strides=1, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 256, 3, strides=1, padding='valid', activation=tf.nn.relu)

        # First layer
        branch = TreeLayer(stage=1, image=x, labels=self.Y[0])
        layer_loss, layer_probabilities, layer_caps = branch(n_outputs=self.Y[0].shape[1])
        self.losses.append(layer_loss)
        self.probabilities.append(layer_probabilities)

        # Other layers
        for i in range(1, self.depth):
            x = tf.stop_gradient(x)
            x = tf.layers.conv2d(x, 256, 3, strides=1, padding='same', activation=tf.nn.relu)
            x_caps = tf.stop_gradient(layer_caps)
            x_caps = tf.reshape(x_caps, [cfg.batch_size, CAPS_SIZE, CAPS_SIZE, -1])
            branch = TreeLayer(stage=i + 1, image=x, labels=self.Y[i], prev_layer_caps=x_caps)
            layer_loss, layer_probabilities, layer_caps = branch(n_outputs=self.Y[i].shape[1])
            self.losses.append(layer_loss)
            self.probabilities.append(layer_probabilities)
