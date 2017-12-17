import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tree_util import build_tree
from cifar_labels import get_batches, get_test_batches, get_predictions, get_accuracy
from treenet import TreeNet

ACCURACY_SAVE_FORMAT = 'plots/accuracy/plot_%s.jpeg'
LOSS_SAVE_FORMAT = 'plots/loss/plot_%s.jpeg'


def build_model(n_classes):
    X = tf.placeholder(dtype=tf.float32, shape=[None, 32 * 32 * 3])
    Y = [tf.placeholder(dtype=tf.float32, shape=[None, n]) for n in n_classes]
    return TreeNet(X, Y)


def train(model, batches, restore=None,
          save=True, plot=True, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        if restore:
            saver.restore(sess, restore)
            print("Model restored.")

        if verbose:
            plt.ion()
            plt.show()
            for i in range(len(batches)):
                print('Batch {} size is {}'.format(i, len(batches[i])))

        losses = []
        accuracies = [[]] * model.depth
        count = 0

        n_iter_per_level = [5000] * model.depth
        save_frequencies = [100, 1000, 2500, 5000, 10000]
        plot_frequency = 100

        # Training loop
        try:
            for level, n_iter in enumerate(n_iter_per_level):
                for i in range(n_iter):
                    fetches = [model.train_ops[level], model.losses[level]]
                    fetches += model.accuracies[:level + 1]

                    batch = batches[level][i % len(batches[level])]

                    feed_dict = {model.X: batch[0]}
                    for j in range(level + 1):
                        feed_dict[model.Y[j]] = batch[1][j]

                    stats = sess.run(fetches, feed_dict=feed_dict)

                    # Save model
                    if save:
                        for k in save_frequencies:
                            if count % k == 0:
                                save_path = saver.save(sess, 'saves/model_{}.cpkt'.format(k))
                                print("Model saved in file: %s" % save_path)

                    # Print and plot
                    if verbose:
                        print('\nStage {}, i: {}'.format(level, i))
                        batch_loss = stats[1]
                        losses.append(batch_loss)
                        print('Loss: {}'.format(batch_loss))

                        for j, acc in enumerate(stats[2:]):
                            accuracies[j].append(acc)
                            print('Layer {} Accuracy: {}'.format(j + 1, acc))

                        # Plot accuracy and error
                        if plot and count % plot_frequency == 0:
                            _plot(accuracies[-1], i, losses)

                    count += 1

        except KeyboardInterrupt:
            if save:
                if verbose:
                    print 'Received keyboard interrupt: saving model...'
                saver.save(sess, 'saves/model_final.cpkt')


def test(model, tree, save_path):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        saver.restore(sess, save_path)
        print("Model restored.")

        test_batches = get_test_batches()
        test_accs = []

        print(len(test_batches))

        for batch in test_batches:
            probabilities = sess.run(model.probabilities, feed_dict={model.X: batch[0]})
            acc = get_accuracy(get_predictions(tree, probabilities), batch[1])
            test_accs.append(acc)
            print(acc)

        print(np.mean(test_accs))


def _plot(accuracies, iteration, losses):
    plt.figure(1)
    plt.plot(accuracies)
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.title('Training Accuracy')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Batch Accuracy')
    plt.savefig(ACCURACY_SAVE_FORMAT % str(iteration), format='png')

    plt.figure(2)
    plt.plot(losses)
    plt.title('Loss')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Batch Loss')
    plt.savefig(LOSS_SAVE_FORMAT % str(iteration), format='png')

    plt.draw()
    plt.pause(0.001)


def main(_):
    tree, n_classes = build_tree('animal_tree.json')
    model = build_model(n_classes)
    train_batches = get_batches(tree)
    train(model, train_batches, save=False, plot=False)


if __name__ == '__main__':
    tf.app.run()
