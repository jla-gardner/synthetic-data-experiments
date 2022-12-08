import numpy as np

__SEED = 42


def set_seed(seed):
    global __SEED
    __SEED = seed


def random(): return np.random.RandomState(seed=__SEED)
def randperm(N): return random().permutation(N)


def shuffle(stuff):
    _order = randperm(len(stuff))
    return stuff[_order]
