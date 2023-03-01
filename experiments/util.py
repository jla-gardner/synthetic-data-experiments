import sys
import functools
from collections.abc import Mapping
from math import ceil
from time import perf_counter
import torch

import numpy as np


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    return x


def mae(a, b):
    return np.abs(to_np(a) - to_np(b)).mean().item()


def test_model(model, X, y, cns=None, batchsize=None):
    """
    Test a model on a dataset
    
    Parameters
    ----------
    model : nn.Module
        model to test
    X : torch.Tensor
        test data (atomic environments)
    y : torch.Tensor
        test labels (local energies)
    cns : np.ndarray
        coordination numbers for each atom
    batchsize : int
        batch size

    Returns
    -------
    dict
        results
    """
    tick = perf_counter()
    with torch.no_grad():
        if batchsize is None:
            yhat = model(X)
        else:
            N = ceil(len(X) / batchsize)
            yhat = None
            for i in range(N):
                _yhat = model(X[i*batchsize:(i+1)*batchsize])
                _yhat = to_np(_yhat)
                if yhat is not None:
                    yhat = np.concatenate([yhat, _yhat])
                else:
                    yhat = _yhat
            yhat = yhat.reshape(-1)

    _time = perf_counter() - tick
    results = dict(
        duration=_time, mae=mae(y, yhat), n_test=len(y),
    )
    if cns is not None:
        results = {
            **results,
            **{f"cn_{i}_mae": mae(y[cns == i], yhat[cns == i])
               for i in (2, 3, 4)}
        }
    return results


def matches_value(value, template_value):
    if isinstance(value, Mapping):
        return matches(value, template_value)
    if callable(template_value):
        return template_value(value)
    return value == template_value


def matches(thing, template):
    for key in set(thing.keys()).union(set(template.keys())):
        if key not in template:
            continue
        if key not in thing:
            return False
        if matches_value(thing[key], template[key]):
            continue
        return False

    return True


def matches_template(t): return functools.partial(matches, template=t)


def identity(x): return x


def parse_passed_kwargs(**types):
    """
    Parse passed keyword arguments from the command line
    
    Example
    -------
    >>> parse_passed_kwargs(n_train=int, n_test=int)
    {'n_train': 100, 'n_test': 100}
    """
    if sys.argv[0] == "./run":
        passed = sys.argv[2:]
    else:
        passed = sys.argv[1:]
    
    raw = dict(p.split("=") for p in passed)
    return {
        k: (types[k] if k in types else identity)(val)
        for k, val in raw.items()
    }
