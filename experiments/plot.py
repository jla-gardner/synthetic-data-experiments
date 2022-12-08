from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def default_plt(func):
    def wrapper(*args, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        return func(*args, ax=ax, **kwargs)
    return wrapper


@default_plt
def log_log(ax=None):
    ax.set_yscale('log')
    ax.set_xscale('log')


@default_plt
def xticks(xs, labels=None, ax=None):
    ax.set_xticks(ticks=xs, labels=xs if labels is None else labels)
    ax.xaxis.set_minor_locator(ticker.NullLocator())


@default_plt
def yticks(ys, labels=None, ax=None):
    ax.set_yticks(ticks=ys, labels=ys if labels is None else labels)
    ax.yaxis.set_minor_locator(ticker.NullLocator())


def indexmin(arr):
    m = arr.min()
    return np.nonzero(arr == m)[0][0]


def bottom_row(points):
    hull = ConvexHull(points)
    xs = points[hull.vertices][:, 0]
    to_keep = np.zeros(len(xs)) == 0

    i = 0
    for a, b, c in zip(np.roll(xs, -1, axis=0), xs, np.roll(xs, 1, axis=0)):
        to_keep[i] = not (a < b == b < c)
        i += 1

    xs = xs[to_keep]
    idxs = np.roll(hull.vertices[to_keep], -indexmin(xs))
    return idxs
