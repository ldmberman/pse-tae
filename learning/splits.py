"""Original paper-compatible 60/20/20 partitions shared by neural and tree models."""
import numpy as np
from sklearn.model_selection import KFold


def partitions(n, fold):
    if fold not in range(1, 6):
        raise ValueError('fold must be between 1 and 5')
    order = np.arange(n)
    np.random.RandomState(1).shuffle(order)
    splits = list(KFold(5, shuffle=False).split(order))
    nv = len(splits[0][1])
    trainval, test = splits[fold - 1]
    return {'train': order[trainval[:-nv]], 'val': order[trainval[-nv:]], 'test': order[test]}
