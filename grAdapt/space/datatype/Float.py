# Python Standard Libraries
import numpy as np

# grAdapt
from .base import Datatype


class Float(Datatype):
    """Float Datatype
    Only the low and upper bound are required to define a Float Datatype.

    >>> from grAdapt.space.datatype import Float
    >>> first_dim = Float(low=10.5, high=20)
    >>> bounds = [first_dim]
    """

    def __init__(self, low, high, prior='uniform'):
        """

        Parameters
        ----------
        low: numeric
        high: numeric
        prior: str
            - if 'uniform' then nothing is changed
            - if 'log-uniform' then samples between low and high are drawn from the log-uniform distribution
        """

        # Exception handling
        if high < low:
            raise ValueError('high must be higher than low.')

        if prior == 'log-uniform':
            if low <= 0:
                raise ValueError('If logarithmic prior is used, then low must be higher than 0.')

        self.low = low
        self.high = high
        self.prior = prior
        self.bound = [low, high]
        self.len = 2
        self.dtype = 'float'

    def __len__(self):
        return self.len

    def __getitem__(self, key):
        return self.bound[key]

    def __setitem__(self, key, value):
        self.bound[key] = value

    def get_value(self, value):
        return self.get_float(value)

    def get_float(self, value):
        return value

    def __repr__(self):
        return 'Float('+str(self.bound)+')'

    def __str__(self):
        return 'Float('+str(self.bound)+')'

    def transform(self, x):
        """

        Parameters
        ----------
        x : numeric

        Returns
        -------
        numeric

        """
        if self.prior == 'log-uniform':
            # Inverse transform sampling using quantile function
            x_normalized = x / self.high
            return self.get_value(self.low * np.power(self.high * 1.0 / self.low, x_normalized))
        else:
            return self.get_value(x)
