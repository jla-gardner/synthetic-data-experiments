from math import log
import pickle

import pyGPs

from experiments.saving import model_path


class SparseGPR:
    """
    Sparse Gaussian process regression model
    
    Parameters
    ----------
    basis : array-like
        locations of inducing points
    noise : float
        Noise level        
    """
    def __init__(self, basis, noise=0.01) -> None:
        self._model = pyGPs.GPR_FITC()
        self._model.setPrior(
            kernel=pyGPs.cov.Poly(d=5), inducing_points=basis
        )
        self._model.setNoise(log(noise))

    def train(self, X, y) -> None:
        """
        Train the model
        """
        mean = pyGPs.mean.Const(y.mean())
        self._model.setPrior(mean=mean)
        self._model.getPosterior(X, y)

    def predict(self, X):
        """
        Predict the output for the given input
        """
        return self._model.predict(X)[0]

    def __call__(self, X):
        return self.predict(X)

    def save(self, name) -> None:
        with open(name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, name) -> 'SparseGPR':
        with open(name, "rb") as f:
            sparse_gpr = pickle.load(f)
        return sparse_gpr


def load_gpr_model(**config):
    """
    Load a sparse GPR model
    """
    return SparseGPR.load(model_path('gpr', config))
