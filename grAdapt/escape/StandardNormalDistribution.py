# Python Standard Libraries
import numpy as np

# grAdapt package
from .base import Escape, inside_bounds, sample_points_bounds


class StandardNormalDistribution(Escape):
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
        x_best = x_train[np.argmin(y_train)]
        cov_matrix = np.eye(len(bounds))

        for i in range(20):
            x_next = np.random.multivariate_normal(x_best, cov_matrix)
            if inside_bounds(bounds, x_next):
                return x_next

        return self.escape_history(bounds, x_train)
