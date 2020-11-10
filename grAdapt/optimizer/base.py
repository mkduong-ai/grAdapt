# Python Standard Libraries
from abc import abstractmethod

# Third party imports
import numpy as np


class Optimizer:
    """Gradient-based Optimization Method Base class

    Parameters
    ----------
    surrogate : surrogate Object
        Surrogate model i.e. Gaussian Process Regression, Random Forrest Regression
    params : list or tuple
        List or tuple of parameters specific for the Optimizer.

    Attributes
    ----------
    self.surrogate : surrogate
    self.params : params


    Examples
    --------
    >>> optimizer = Optimizer(surrogate, params)
    >>> optimizer.run(np.array([1, 2, 3.4]), 100, [x_train])
    """

    def __init__(self, surrogate, params):
        self.surrogate = surrogate
        self.params = params

    def __call__(self, surrogate, params=None):
        self.surrogate = surrogate

        if params is not None:
            self.params = params

    def set_params(self, params):
        self.params = params

    @abstractmethod
    def run(self, xp, num_iters, surrogate_grad_params):
        """Run the optimizer from the starting point xp with n_iters steps

        Parameters
        ----------
        xp : 1D array-like (d,)
            Starting point for the Optimizer.
        num_iters : int
            Number of steps to be made.
        surrogate_grad_params : list or tuple
            List or tuple of parameters for the surrogate model. Different surrogate models require different parameters
            to evaluate the gradient.

        Returns
        -------
        x_next : 1D array-like

        """
        raise NotImplementedError