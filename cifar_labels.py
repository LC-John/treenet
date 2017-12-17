import copy

import numpy as np

from config import cfg
from tree_util import *

np.set_printoptions(threshold=np.nan)

DATA_DIR = 'cifar-100-python/'
batch_size = cfg.batch_size


# noinspection PyShadowingNames
def get_test_batches():
    test_set = unpickle(DATA_DIR + 'test')
    test_images = test_set['data']
    test_labels = test_set['fine_labels']

    batches = []
    for i in range(0, 10000, batch_size):
        if i <= 10000 - batch_size:
            img_batch = test_images[i:i + batch_size]
            lab_batch = test_labels[i:i + batch_size]
            batches.append((img_batch, lab_batch))

    return batches


# noinspection PyShadowingNames
def get_batches(tree, train=True):
    stage_lists = get_leaves_per_level(tree)
    stage_labels = get_path_to_labels(tree)

    data_set = unpickle(DATA_DIR + ('train' if train else 'test'))
    rel_indices = get_images(data_set['data'], data_set['fine_labels'])

    stage_batches = []
    batch_lab_arrs = []

    for n in range(len(stage_lists)):
        lab_arr = np.zeros((batch_size, len(stage_lists[n])))
        batch_lab_arrs.append(copy.deepcopy(lab_arr))

    for n in range(len(stage_lists)):
        sample_set = []
        sample_set_labels = []
        batch_order = []
        stage_list = stage_lists[n]
        max_length = 1

        n_lists = len(stage_list)

        for i in range(len(stage_list)):
            set = []
            set_labels = []

            for index in stage_list[i]:
                set += rel_indices[index]
                set_labels += [stage_labels[index][0:n + 1]] * 500

            sample_set_labels.append(copy.deepcopy(set_labels))
            sample_set.append(copy.deepcopy(set))

            batch_order.append(np.arange(len(sample_set[i])))
            np.random.shuffle(batch_order[i])

            if len(sample_set[i]) >= max_length:
                max_length = len(sample_set[i])
                max_arg = i

        class_size = int(np.ceil(batch_size / len(sample_set)))

        for i in range(n_lists):
            batch_order[i] = np.tile(batch_order[i], int(np.ceil(max_length / float(len(batch_order[i])))))

        class_order = []
        batches = []

        for i in range(0, max_length * len(sample_set), class_size * n_lists):
            for j in range(n_lists):
                class_order += [j] * class_size

        curr_class_img = np.zeros(n_lists, dtype=np.int32)
        current = 0
        done = False

        while not done:
            images = []
            batch_labels = copy.deepcopy(batch_lab_arrs)
            for j in range(batch_size):
                curr_class = class_order[current]
                curr_batch_img = batch_order[curr_class][curr_class_img[curr_class]]
                images.append(sample_set[curr_class][curr_batch_img])
                for k in range(len(sample_set_labels[curr_class][curr_batch_img])):
                    batch_labels[k][j, sample_set_labels[curr_class][curr_batch_img][k]] = 1
                current += 1
                curr_class_img[curr_class] += 1
                if curr_class_img[curr_class] == max_length:
                    if curr_class == max_arg:
                        final_imgs = batch_order[curr_class][int(max_length - (batch_size - len(images))):max_length]
                        final_index = j + 1
                        for img in final_imgs:
                            images.append(sample_set[curr_class][img])
                            for k in range(len(sample_set_labels[curr_class][img])):
                                batch_labels[k][final_index, sample_set_labels[curr_class][img][k]] = 1
                            final_index += 1
                        done = True
                        break
                    curr_class_img[curr_class] = 0
            batches.append((np.asarray(images), copy.deepcopy(batch_labels)))

        stage_batches.append(copy.deepcopy(batches))

    return stage_batches


# noinspection PyShadowingNames
def get_images(images, labels):
    result = [[] for _ in range(100)]
    for i in range(len(labels)):
        result[labels[i]].append(images[i])
    return result


# noinspection PyShadowingNames
def get_predictions(tree, probabilities):
    n = len(probabilities[0])
    queue = deque([tree])
    predictions = deque([np.zeros((n, 0))])
    result = np.zeros((n, 100))
    level = 0

    while queue:
        count = 0
        for i in range(len(queue)):
            node = queue.popleft()
            prior = predictions.popleft()
            if not node.children:
                result[:, node.true_label] = np.mean(prior, axis=1)
                continue
            for child in node.children:
                queue.append(child)
                prediction = np.c_[prior, probabilities[level][:, count]]
                predictions.append(prediction)
                count += 1
        level += 1

    return result


# noinspection PyShadowingNames
def get_accuracy(predictions, test_labels):
    predictions = np.argmax(predictions, axis=1)
    acc = [int(predictions[i] == test_labels[i]) for i in range(batch_size)]
    return np.mean(acc)
