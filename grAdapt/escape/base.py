# Python Standard Libraries
# import numpy as np
# decorators
from abc import abstractmethod

# grAdapt package
from grAdapt.utils.sampling import *
from grAdapt.sampling.equidistributed import MaximalMinDistance


class Escape:
    """Escape Base class

    Parameters
    ----------
    self.surrogate : surrogate object

    Attributes
    ----------


    Examples
    --------
    """

    def __init__(self, surrogate, sampling_method=None):
        self.surrogate = surrogate
        if sampling_method is None:
            self.sampling_method = MaximalMinDistance()
        else:
            self.sampling_method = sampling_method

    def __call__(self, surrogate):
        self.surrogate = surrogate

    def escape_history(self, bounds, x_train):
        """Escape considering history of points

        Parameters
        ----------
        bounds : list of tuples.
            Each tuple in the list defines the bounds for the corresponding variable
            Example: [(1, 2), (2, 3), (-1, 4)...]
        x_train : array-like (n, dim)
            n points with dim dimensions

        Returns
        -------
        array-like (1, dim):
            Returns a 2D array. dim is the dimension of a single point
            row corresponds to a single point.
            column corresponds to a dimension.

        """
        return self.sampling_method.sample(bounds=bounds, n=1, x_history=x_train)

    @abstractmethod
    def get_point(self, x_train, y_train, iteration, bounds):
        return NotImplementedError
