import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cifar_labels import get_batches
from treenet import TreeNet


def train():
    X = tf.placeholder(dtype=tf.float32, shape=[None, 32 * 32 * 3])
    Y_1 = tf.placeholder(dtype=tf.float32, shape=[None, 6])
    Y_2 = tf.placeholder(dtype=tf.float32, shape=[None, 12])
    Y_3 = tf.placeholder(dtype=tf.float32, shape=[None, 33])
    Y_4 = tf.placeholder(dtype=tf.float32, shape=[None, 34])
    Y_5 = tf.placeholder(dtype=tf.float32, shape=[None, 25])
    Y_6 = tf.placeholder(dtype=tf.float32, shape=[None, 12])
    tree_depth = tf.placeholder(dtype=tf.int32)
    model = TreeNet(X, Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, tree_depth)

    with tf.Session() as sess:
        batches = get_batches()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        losses = []
        accs_1 = []
        accs_2 = []
        accs_3 = []
        accs_4 = []
        accs_5 = []
        accs_6 = []
        count = 0
        plt.ion()
        plt.show()
        max_iters = 6000
        t_vars = []
        for i in range(max_iters + 1):
            print('Iteration :{}'.format(i))
            if i < 1000:
                if count >= len(batches[0]):
                    count = 0
                _, batch_loss, acc_1 = sess.run([model.train_op, model.total_loss, model.accuracy_1],
                                                feed_dict={X: batches[0][count][0], Y_1: batches[0][count][1][0],
                                                           Y_2: batches[0][count][1][1], Y_3: batches[0][count][1][2],
                                                           Y_4: batches[0][count][1][3], Y_5: batches[0][count][1][4],
                                                           Y_6: batches[0][count][1][5], tree_depth: 1})
                losses.append(batch_loss)
                accs_1.append(acc_1)
                print('Loss :{}'.format(batch_loss))
                print('Layer 1 Accuracy: {}'.format(acc_1))
                plot_accs = accs_1
                count += 1
            elif i < 2000:
                if count >= len(batches[1]):
                    count = 0
                _, batch_loss, acc_1, acc_2 = sess.run(
                    [model.train_op, model.total_loss, model.accuracy_1, model.accuracy_2],
                    feed_dict={X: batches[1][count][0], Y_1: batches[1][count][1][0],
                               Y_2: batches[1][count][1][1],
                               Y_3: batches[1][count][1][2],
                               Y_4: batches[1][count][1][3],
                               Y_5: batches[1][count][1][4],
                               Y_6: batches[1][count][1][5], tree_depth: 2})
                losses.append(batch_loss)
                accs_1.append(acc_1)
                accs_2.append(acc_2)
                print('Loss :{}'.format(batch_loss))
                print('Layer 1 Accuracy: {}'.format(acc_1))
                print('Layer 2 Accuracy: {}'.format(acc_2))
                plot_accs = accs_2
                count += 1
            elif i < 3000:
                if count >= len(batches[2]):
                    count = 0
                _, batch_loss, acc_1, acc_2, acc_3 = sess.run(
                    [model.train_op, model.total_loss, model.accuracy_1, model.accuracy_2, model.accuracy_3],
                    feed_dict={X: batches[2][count][0], Y_1: batches[2][count][1][0],
                               Y_2: batches[2][count][1][1], Y_3: batches[2][count][1][2],
                               Y_4: batches[2][count][1][3], Y_5: batches[2][count][1][4],
                               Y_6: batches[2][count][1][5], tree_depth: 3})

                losses.append(batch_loss)
                accs_1.append(acc_1)
                accs_2.append(acc_2)
                accs_3.append(acc_3)
                print('Loss :{}'.format(batch_loss))
                print('Layer 1 Accuracy: {}'.format(acc_1))
                print('Layer 2 Accuracy: {}'.format(acc_2))
                print('Layer 3 Accuracy: {}'.format(acc_3))
                plot_accs = accs_3
                count += 1

            elif i < 4000:
                if count >= len(batches[3]):
                    count = 0
                _, batch_loss, acc_1, acc_2, acc_3, acc_4 = sess.run(
                    [model.train_op, model.total_loss, model.accuracy_1, model.accuracy_2, model.accuracy_3,
                     model.accuracy_4],
                    feed_dict={X: batches[3][count][0], Y_1: batches[3][count][1][0],
                               Y_2: batches[3][count][1][1], Y_3: batches[3][count][1][2],
                               Y_4: batches[3][count][1][3], Y_5: batches[3][count][1][4],
                               Y_6: batches[3][count][1][5], tree_depth: 4})

                losses.append(batch_loss)
                accs_1.append(acc_1)
                accs_2.append(acc_2)
                accs_3.append(acc_3)
                accs_4.append(acc_4)
                print('Loss :{}'.format(batch_loss))
                print('Layer 1 Accuracy: {}'.format(acc_1))
                print('Layer 2 Accuracy: {}'.format(acc_2))
                print('Layer 3 Accuracy: {}'.format(acc_3))
                print('Layer 4 Accuracy: {}'.format(acc_4))
                plot_accs = accs_4
                count += 1

            elif i < 5000:
                if count >= len(batches[4]):
                    count = 0
                _, batch_loss, acc_1, acc_2, acc_3, acc_4, acc_5 = sess.run(
                    [model.train_op, model.total_loss, model.accuracy_1, model.accuracy_2, model.accuracy_3,
                     model.accuracy_4, model.accuracy_5],
                    feed_dict={X: batches[4][count][0], Y_1: batches[4][count][1][0],
                               Y_2: batches[4][count][1][1], Y_3: batches[4][count][1][2],
                               Y_4: batches[4][count][1][3], Y_5: batches[4][count][1][4],
                               Y_6: batches[4][count][1][5], tree_depth: 5})

                losses.append(batch_loss)
                accs_1.append(acc_1)
                accs_2.append(acc_2)
                accs_3.append(acc_3)
                accs_4.append(acc_4)
                accs_5.append(acc_5)
                print('Loss :{}'.format(batch_loss))
                print('Layer 1 Accuracy: {}'.format(acc_1))
                print('Layer 2 Accuracy: {}'.format(acc_2))
                print('Layer 3 Accuracy: {}'.format(acc_3))
                print('Layer 4 Accuracy: {}'.format(acc_4))
                print('Layer 5 Accuracy: {}'.format(acc_5))
                plot_accs = accs_5
                count += 1

            elif i < 6000:
                if count >= len(batches[5]):
                    count = 0
                _, batch_loss, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6 = sess.run(
                    [model.train_op, model.total_loss, model.accuracy_1, model.accuracy_2, model.accuracy_3,
                     model.accuracy_4, model.accuracy_5, model.accuracy_6],
                    feed_dict={X: batches[4][count][0], Y_1: batches[4][count][1][0],
                               Y_2: batches[4][count][1][1], Y_3: batches[4][count][1][2],
                               Y_4: batches[4][count][1][3], Y_5: batches[4][count][1][4],
                               Y_6: batches[4][count][1][5], tree_depth: 5})

                losses.append(batch_loss)
                accs_1.append(acc_1)
                accs_2.append(acc_2)
                accs_3.append(acc_3)
                accs_4.append(acc_4)
                accs_5.append(acc_5)
                accs_6.append(acc_6)
                print('Loss :{}'.format(batch_loss))
                print('Layer 1 Accuracy: {}'.format(acc_1))
                print('Layer 2 Accuracy: {}'.format(acc_2))
                print('Layer 3 Accuracy: {}'.format(acc_3))
                print('Layer 4 Accuracy: {}'.format(acc_4))
                print('Layer 5 Accuracy: {}'.format(acc_5))
                print('Layer 6 Accuracy: {}'.format(acc_6))
                plot_accs = accs_6
                count += 1

            if i % 250 == 0:
                save_path = saver.save(sess, 'C:/Users/Christian/Desktop/treenet/layer_0/model.cpkt')
                print("Model saved in file: %s" % save_path)

            if i % 50 == 0:
                plt.figure(1)
                plt.plot(losses)

                plt.figure(2)
                plt.plot(plot_accs)
                plt.yticks(np.arange(0, 1.05, 0.05))

                plt.draw()
                plt.pause(0.001)

            if i == max_iters:
                plt.show()


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
