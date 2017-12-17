import json
from collections import deque

from util import unpickle


class Node(object):
    def __init__(self, name, children=None, true_label=-1):
        self.name = name
        self.children = children
        self.index = -1
        self.true_label = true_label

    def __str__(self):
        s = self.name
        if self.children:
            s += ' -> ' + ', '.join([c.name for c in self.children])
        return s


def build_tree(filename):
    """
    Build the tree of classes from a json specification
    """
    with open(filename, 'rb') as json_file:
        json_data = json.load(json_file)

    meta = unpickle('cifar-100-python/meta')
    tree = _build_tree(json_data, meta['fine_label_names'])[0]

    # Get the number of classes per level
    n_classes = []
    queue = deque(tree.children)
    while queue:
        n_classes.append(len(queue))
        for i in range(len(queue)):
            node = queue.popleft()
            node.index = i
            if node.children:
                for child in node.children:
                    queue.append(child)

    return tree, n_classes


def _build_tree(json_dict, labels):
    children = []
    for key in json_dict:
        value = json_dict[key]
        if not value:
            children.append(Node(key, true_label=labels.index(key)))
        else:
            children.append(Node(key, _build_tree(value, labels)))
    return children


def get_leaves(root, leaves=None):
    """
    Get a list with the labels of all leaves in the tree
    """
    if leaves is None:
        leaves = []

    if not root.children:
        leaves.append(root.true_label)
    else:
        for child in root.children:
            get_leaves(child, leaves)

    return leaves


def get_path_to_labels(node):
    """
    Get the indices at every level on the path to each class
    """
    queue = deque(node.children)
    paths = deque([[]] * len(node.children))
    path_to_labels = [[]] * 100

    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            path = paths.popleft()
            if not node.children:
                path_to_labels[node.true_label] = path + [node.index]
                continue
            for child in node.children:
                queue.append(child)
                paths.append(path + [node.index])

    return path_to_labels


def get_leaves_per_level(node):
    """
    Get the leaves under each class at every level
    """
    leaves_per_level = []
    _get_level_leaves(node, leaves_per_level, 0)
    return leaves_per_level[1:]


def _get_level_leaves(node, leaves_per_level, level):
    if len(leaves_per_level) <= level:
        leaves_per_level.append([])

    if not node.children:
        leaf = [node.true_label]
        leaves_per_level[level].append(leaf)
        return leaf

    leaves = []
    for child in node.children:
        leaves += _get_level_leaves(child, leaves_per_level, level + 1)

    leaves_per_level[level].append(leaves)
    return leaves
