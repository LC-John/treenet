import sys


# noinspection PyDefaultArgument
def flatten(ls, acc=None):
    """
    Flatten ND list to 1D

    An accumulator list may be passed where the flattened
    list elements will be appended to
    """
    if acc is None:
        acc = []

    if type(ls) is not list:
        acc.append(ls)
    else:
        for item in ls:
            flatten(item, acc=acc)

    return acc


def unpickle(filename):
    """
    Unpickle the given file and return the data
    """
    with open(filename, mode='rb') as f:
        if (2, 7) <= sys.version_info < (3,):
            import cPickle
            return cPickle.load(f)
        else:
            import pickle
            return pickle.load(f, encoding='latin1')
