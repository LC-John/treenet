import json
import os

from util import *


def _build_tree(tree, labels):
    children = []
    for key in tree:
        value = tree[key]
        if not value:
            children.append([labels.index(key)])
        else:
            children.append(_build_tree(value, labels))
    return children


def build_tree():
    with open('tree.json', 'rb') as json_file:
        json_data = json.load(json_file)
        meta = unpickle(os.path.abspath('cifar-100-python/meta'))
        labels = meta[b'fine_label_names']
        tree = _build_tree(json_data, labels)[0]

        # Sanity check
        assert sorted(flatten(tree)) == [i for i in range(100)]

        return tree
