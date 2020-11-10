# Python Standard Libraries
# import numpy as np

# grAdapt package
from .base import Escape


class UniformDistribution(Escape):
    """Standard Multivariate Normal Distribution
    is added to the current best position

    """

    def __init__(self, surrogate, sampling_method=None):
        super().__init__(surrogate, sampling_method)

    def get_point(self, x_train, y_train, iteration, bounds):
        """
        Parameters
        ----------
        self : self object
        x_train : array-like shape (n, d)
        y_train : array-like shape (n,)
        iteration : integer
        bounds: list of 2-tuples

        Returns
        -------
        array-like (d,)
        """
        return self.escape_history(bounds, x_train)
