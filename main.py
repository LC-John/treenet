import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cifar_labels import get_batches
from treenet import TreeNet

MAX_ITER = 6000


def train():
    # Construct model
    X = tf.placeholder(dtype=tf.float32, shape=[None, 32 * 32 * 3])
    Y_1 = tf.placeholder(dtype=tf.float32, shape=[None, 6])
    Y_2 = tf.placeholder(dtype=tf.float32, shape=[None, 12])
    Y_3 = tf.placeholder(dtype=tf.float32, shape=[None, 33])
    Y_4 = tf.placeholder(dtype=tf.float32, shape=[None, 34])
    Y_5 = tf.placeholder(dtype=tf.float32, shape=[None, 25])
    Y_6 = tf.placeholder(dtype=tf.float32, shape=[None, 12])

    tree_depth = tf.placeholder(dtype=tf.int32)
    model = TreeNet(X, Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, tree_depth)

    # Train model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        batches = get_batches()
        saver = tf.train.Saver()

        losses = []
        accuracies = [[]] * tree_depth

        plt.ion()
        plt.show()

        count = 0
        for i in range(MAX_ITER):
            print('Iteration :{}'.format(i))

            # Stage specific code
            stage = i / 1000
            batch = batches[stage]

            fetches = [model.train_op, model.total_loss]
            fetches += model.accuracies[:stage + 1]
            results = sess.run(fetches, feed_dict={X: batch[count][0],
                                                   Y_1: batch[count][1][0],
                                                   Y_2: batch[count][1][1],
                                                   Y_3: batch[count][1][2],
                                                   Y_4: batch[count][1][3],
                                                   Y_5: batch[count][1][4],
                                                   Y_6: batch[count][1][5],
                                                   tree_depth: stage + 1})
            batch_loss = results[1]
            batch_accuracies = results[2:]

            losses.append(batch_loss)
            print('Loss :{}'.format(batch_loss))
            for j in range(stage + 1):
                accuracies[j].append(batch_accuracies[j])
                print('Layer 1 Accuracy: {}'.format(batch_accuracies[j]))

            plot_accs = batch_accuracies[-1]
            count = (count + 1) % len(batch)

            # Save model and print results
            if i % 250 == 0:
                save_path = saver.save(sess, 'save/model.cpkt')
                print("Model saved in file: %s" % save_path)

            if i % 50 == 0:
                plt.figure(1)
                plt.plot(losses)

                plt.figure(2)
                plt.plot(plot_accs)
                plt.yticks(np.arange(0, 1.05, 0.05))

                plt.draw()
                plt.pause(0.001)

            if i == MAX_ITER:
                plt.show()


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
