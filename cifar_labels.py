import copy
import os

import numpy as np

from build_tree import build_tree
from config import cfg

DATA_DIR = 'cifar-100-python/'
batch_size = cfg.batch_size


def unpickle(file):
    import cPickle as pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


train_set = unpickle(os.path.abspath(DATA_DIR + 'train'))
train_meta = unpickle(os.path.abspath(DATA_DIR + 'meta'))
test_set = unpickle(os.path.abspath(DATA_DIR + 'test'))

train_images = train_set[b'data']
train_labels = train_set[b'fine_labels']
coarse_labels = train_set[b'coarse_labels']
filenames = train_set[b'filenames']

test_images = test_set[b'data']
test_labels = test_set[b'fine_labels']

# train_images = np.reshape(train_images, (50000, 3, 32, 32))
# train_images = np.transpose(train_images, (0, 2, 3, 1))

train_labels_names = train_meta[b'fine_label_names']


def get_batches(train=True):
    stage_lists, stage_labels = get_stage_lists(build_tree(),
                                                [[], [], [], [], [], [], []],
                                                [[], [], [], [], [], [], []])

    if train:
        rel_indices = get_images(train_images, train_labels)
    else:
        rel_indices = get_images(test_images, test_labels)

    stage_batches = []
    batch_lab_arrs = []

    for n in range(len(stage_lists)):
        lab_arr = np.zeros((batch_size, len(stage_lists[n])))
        batch_lab_arrs.append(copy.deepcopy(lab_arr))

    for n in range(len(stage_lists)):
        batch_labels = copy.deepcopy(batch_lab_arrs)
        sample_set = []
        stage_list = stage_lists[n]
        stage_label = stage_labels[n]
        max_length = 1

        for i in range(len(stage_list)):
            if len(stage_list[i]) > max_length:
                max_length = len(stage_list[i])
            set = []
            for index in stage_list[i]:
                set += rel_indices[str(index)]
            sample_set.append(copy.deepcopy(set))
        n_list_items = float(len(stage_list))
        class_size = int(np.floor(batch_size / n_list_items))
        batches = []
        num_batches = int(max_length * 500 / float(batch_size)) * 2

        for j in range(num_batches):
            batch_images = []
            start = 0
            end = start + class_size
            diff = batch_size - class_size * len(stage_list)
            for i in range(len(stage_list)):
                if diff > 0:
                    end += 1
                    diff -= 1
                np.random.shuffle(sample_set[i])
                batch_images += sample_set[i][0:end - start]
                for k in range(len(stage_label[i])):
                    batch_labels[k][start:end, stage_label[i][k]] = 1
                start = end
                end += class_size
            batches.append((np.asarray(batch_images), batch_labels))
        stage_batches.append(batches)

    # Make sure one hot encoding is in place
    for stage_num in range(len(stage_batches)):
        fifth_mini_batch = stage_batches[stage_num][4]
        _, y_5 = fifth_mini_batch
        for layer in range(stage_num + 1):
            assert((np.sum(y_5[layer], axis=1, keepdims=True) == np.ones((batch_size, 1))).all())

    return stage_batches


def flatten_list(nested_list, curr_list):
    if type(nested_list) is not list:
        curr_list.append(nested_list)
        return
    for item in nested_list:
        flatten_list(item, curr_list)
    return curr_list


def get_stage_lists(parent_list, stage_list, label_list, prev_index=[], depth=-1):
    if depth >= 0:
        curr_index = prev_index + [len(stage_list[depth])]
        stage_list[depth].append(flatten_list(parent_list, []))
        label_list[depth].append(curr_index)
    else:
        curr_index = prev_index

    if len(parent_list) == 1:
        return

    for i in range(len(parent_list)):
        get_stage_lists(parent_list[i], stage_list, label_list,
                        prev_index=curr_index, depth=depth + 1)

    return stage_list, label_list


def get_images(images, labels):
    class_indices = {}
    for i in range(len(labels)):
        if str(labels[i]) in class_indices:
            class_indices[str(labels[i])].append(images[i])
        else:
            class_indices[str(labels[i])] = [images[i]]
    return class_indices


if __name__ == '__main__':
    batches = get_batches()
