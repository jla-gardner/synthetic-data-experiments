import numpy as np
from ase.io import read
from locache import persist
from quippy.descriptors import Descriptor
from ase import Atoms
from ase.neighborlist import neighbor_list

from experiments.random import randperm, shuffle
from experiments.saving import _top_dir


def soap_descriptor(**kwargs):
    """
    Generate a SOAP descriping function

    Example
    -------
    >>> soap = soap_descriptor(n_max=3, l_max=3, cutoff=3.7, atom_sigma=0.5)
    >>> soap(Atoms('1C'))
    """
    _desc = Descriptor(
        "soap " + " ".join(f"{k}={v}" for k, v in kwargs.items())
    )
    return lambda atoms: _desc.calc(atoms)['data']


@persist(name="data")
def __get_structures(n_max, l_max):
    """
    Read the dataset and calculate SOAP descriptors
    """
    structures = read(_top_dir / "all.extxyz", index=":")
    soap = soap_descriptor(
        n_max=n_max, l_max=l_max, cutoff=3.7, atom_sigma=0.5
    )
    for s in structures:
        s.arrays['soap'] = soap(s)
        s.arrays['coord_nums'] = neighbour_counts(s)
    return structures


def descriptor_length(n_max, l_max):
    """
    Get the length of the SOAP descriptor for a given n_max and l_max
    """
    _c = Atoms('1C')
    soap = soap_descriptor(n_max=n_max, l_max=l_max,
                           cutoff=3.7, atom_sigma=0.5)
    return soap(_c).size


def neighbour_counts(atoms: Atoms, cutoff=1.8):
    """
    Get the number of neighbours for each atom in the structure
    ignoring self-interactions

    Parameters
    ----------
    atoms : Atoms
        The structure
    cutoff : float
        The cutoff distance for neighbours

    Returns
    -------
    counts : np.ndarray
        The number of neighbours for each atom
    """

    nl = neighbor_list("i", atoms, cutoff, self_interaction=True)
    _, counts = np.unique(nl, return_counts=True)
    return counts - 1


def data_for(fold, folds, n_max, l_max, n_train, equal_cns=False):
    """
    Get the training and test data for a given fold.

    The validation procedure used here is as follows:

    0. All structures are associated with a given id, that is unique
        for each trajectory. Several correlated structures are present 
        for each id, and so we need to make sure that these all appear
        in the same fold to avoid data leakage.
    1. We shuffle the ids, and then split them into k folds.
    2. We split the structures into the k folds based on their ids.
    3. We calculate the X (soaps), y (energies) and cns for each fold.
    4. We shuffle the X, y and cns for each fold to ensure that if less than
        the full fold is used, we have a representative sample across the
        dataset.
    5. We concatenate the relevant folds to make train and test sets.

    If equal_cns is True, then we remove atomic environments from each fold
    until we have equal numbers of atoms with each coordination number.

    Parameters
    ----------
    fold : int
        The fold to use
    folds : int
        The number of folds (the k in k-fold cross validation)
    n_max : int
        The n_max parameter for the SOAP descriptor
    l_max : int
        The l_max parameter for the SOAP descriptor
    n_train : int
        The number of training structures to use
    equal_cns : bool
        Whether to use the same number of structures with each coordination
        number

    Returns
    -------
    x_train : np.ndarray
        The training data
    y_train : np.ndarray
        The training labels
    x_test : np.ndarray
        The test data
    y_test : np.ndarray
        The test labels
    cns_train : np.ndarray
        The coordination numbers for the training structures
    cns_test : np.ndarray
        The coordination numbers for the test structures
    """


    structures = __get_structures(n_max, l_max)

    ids = np.array([s.info['id'] for s in structures])

    unique_ids = np.unique(ids)
    N = len(unique_ids)

    # shuffle ids and generate train-test split
    _shuffle = randperm(N)
    _shuffle = np.roll(_shuffle, fold * N // folds)
    split = (folds-1)*N//folds
    train_ids = unique_ids[_shuffle[:split]]

    train = np.isin(ids, train_ids)
    test = ~train
    x = np.array([s.arrays['soap'] for s in structures])
    y = np.array([
        s.arrays['gap17_energy'].reshape(-1, 1)
        for s in structures
    ])
    cns = np.array([s.arrays['coord_nums'].reshape(-1, 1) for s in structures])

    del structures

    if equal_cns:
        cns_train = shuffle(np.vstack(cns[train])).reshape(-1)
        allowed = np.zeros(len(cns_train)) == 1
        cn_2s = (cns_train == 2).sum()
        _N = min(n_train//3, cn_2s)
        for i in (2, 3, 4):
            cn_i = cns_train == i
            idxs = np.nonzero(cn_i)[0][:_N]
            allowed[idxs] = True

        def process(a):
            return (
                shuffle(np.vstack(a[train]))[allowed][:n_train],
                shuffle(np.vstack(a[test]))
            )
    else:
        def process(a):
            # concatenate the data into one array
            # and shuffle the resulting atomic environments
            return (
                shuffle(np.vstack(a[train]))[:n_train],
                shuffle(np.vstack(a[test]))
            )

    (x_train, x_test), (y_train, y_test), (cn_train, cn_test) = \
        process(x), process(y), process(cns)

    return x_train, y_train, cn_train.reshape(-1), x_test, y_test, cn_test.reshape(-1)
