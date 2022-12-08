from math import exp, log
from random import random


def choose_uniform(min, max):
    return min + (max - min) * random()


def choose_log(min, max):
    choice = choose_uniform(log(min), log(max))
    return exp(choice)


def random_choice(options):
    i = int(choose_uniform(0, len(options)))
    return options[i]


_methods = dict(log=choose_log, uniform=choose_uniform, choice=random_choice)


def pluck(dict, *args):
    return (
        *(dict[arg] if arg in dict else None for arg in args),
        {arg: dict[arg] for arg in dict.keys() if arg not in args}
    )


def sample(spec):
    results = {}
    for name, options in spec.items():
        _method, _type, kwargs = pluck(options, 'method', 'type')
        value = _methods[_method](**kwargs)
        results[name] = _type(value) if _type is not None else value
    return results
