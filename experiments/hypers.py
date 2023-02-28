from math import exp, log
from random import random


def choose_uniform(min, max):
    """
    Choose a random number between min and max
    """
    return min + (max - min) * random()


def choose_log(min, max):
    """
    Choose a random number between min and max, with a log distribution
    """
    choice = choose_uniform(log(min), log(max))
    return exp(choice)


def random_choice(options):
    """
    Choose a random element from a list
    """
    i = int(choose_uniform(0, len(options)))
    return options[i]


_methods = dict(log=choose_log, uniform=choose_uniform, choice=random_choice)


def pluck(dict, *args):
    """
    Pluck values from a dictionary, returning a tuple of the values and a
    dictionary of the remaining values
    """
    return (
        *(dict[arg] if arg in dict else None for arg in args),
        {arg: dict[arg] for arg in dict.keys() if arg not in args}
    )


def sample(spec):
    """
    Sample from a specification

    Parameters
    ----------
    spec : dict
        A dictionary of parameter names and options

    Returns
    -------
    results : dict
        A dictionary of sampled parameter names and values
    
    Examples
    --------
    >>> sample({'a': {'method': 'uniform', 'min': 0, 'max': 1}})
    {'a': 0.123}
    """
    results = {}
    for name, options in spec.items():
        _method, _type, kwargs = pluck(options, 'method', 'type')
        value = _methods[_method](**kwargs)
        results[name] = _type(value) if _type is not None else value
    return results
