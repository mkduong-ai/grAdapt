# Third party imports
import numpy as np

# Package imports
from .base import Surrogate


class NoGradient(Surrogate):
    """NoGradient Surrogate
    Returns a gradient of zero entries
    Optimizer will converge and Model will exploit
    """

    def __init__(self, **surrogate_params):
        super().__init__(None, surrogate_params)

    def fit(self, X, y):
        """Fit training data (X, y)
        Parameters
        ----------
        X : array-like (n, d)
        y : array-like (n,)


        Returns
        -------
        None

        """
        pass

    def predict(self, x_predict, return_std=False):
        """Predict y value based on x_predict
        Parameters
        ----------
        x_predict : array-like (n, d)
        return_std : array-like (n,)
            returns the standard deviation of the posterior mean


        Returns
        -------
        array-like (n,)

        """
        if return_std:
            return np.zeros_like(x_predict), np.zeros_like(x_predict)
        else:
            return np.zeros_like(x_predict)

    def eval_gradient(self, x, surrogate_grad_params):
        """Does not evaluate the gradient
        Parameters
        ----------
        x: 1D array-like (d,)
        surrogate_grad_params: list
            surrogate_grad_params[0]: whole training data shape (m, d)


        Returns
        -------
        grad : np.zeros_like(x)
            same shape as x
        """
        return np.zeros_like(x)
