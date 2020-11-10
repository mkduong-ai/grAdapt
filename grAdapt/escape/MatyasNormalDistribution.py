# Python Standard Libraries
import numpy as np

# grAdapt package
from .base import Escape, inside_bounds, bounds_range_ndim


class MatyasNormalDistribution(Escape):
    """Evolution Strategy (1+1) by Matyas
    Sample till a better point is found given the surrogate model
    Does not work well on higher dimensions > 10 and should be used with caution
    Performance depends on surrogate model
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
        cov_matrix = np.diag((bounds_range_ndim(bounds) / 6) ** 2)

        for i in range(20):
            x_new = np.random.multivariate_normal(x_best, cov_matrix)
            if self.surrogate.predict(x_new) < np.min(y_train) and inside_bounds(bounds, x_new):
                return x_new

        return self.escape_history(bounds, x_train)
