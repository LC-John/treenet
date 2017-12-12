import cPickle


def flatten(ls, acc=[]):
    """
    Flatten ND list to 1D

    An accumulator list may be passed where the flattened
    list elements will be appended to
    """
    if type(ls) is not list:
        acc.append(ls)
        return acc

    for item in ls:
        flatten(item, acc=acc)

    return acc


def unpickle(filename):
    """ Unpickle the given file and return the data """
    with open(filename, mode='rb') as f:
        data = cPickle.load(f)

    return data
