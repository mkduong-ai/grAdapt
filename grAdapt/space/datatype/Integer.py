# Python Standard Libraries
import numpy as np

# grAdapt
from .base import Datatype


class Integer(Datatype):
    """Integer Datatype
    Only the low and upper bound are required to define a Integer Datatype.

    >>> from grAdapt.space.datatype import Integer
    >>> first_dim = Integer(low=10, high=100)
    >>> bounds = [first_dim]
    """

    def __init__(self, low, high, prior='uniform'):
        """

        Parameters
        ----------
        low: int
        high: int
        prior: str
            - if 'uniform' then nothing is changed
            - if 'log-uniform' then samples between low and high are drawn from the log-uniform distribution
        """

        # Exception handling
        if high < low:
            raise ValueError("high must be higher than low.")

        if prior == 'log-uniform':
            if low <= 0:
                raise ValueError('If logarithmic prior is used, then low must be higher than 0.')

        self.low = low
        self.high = high
        if self.low % 1 > 0 or self.high % 1 > 0:
            raise ValueError('low and high must be integers.')
        self.prior = prior
        self.bound = [self.low, self.high]
        self.len = 2
        self.dtype = 'int'

    def __len__(self):
        return self.len

    def __getitem__(self, key):
        return self.bound[key]

    def __setitem__(self, key, value):
        self.bound[key] = value
    
    def get_value(self, value):
        return self.get_integer(value)
    
    def get_integer(self, value):
        return int(np.round(value))

    def __repr__(self):
        return 'Integer('+str(self.bound)+')'

    def __str__(self):
        return 'Integer('+str(self.bound)+')'

    def transform(self, x):
        """Single value to transform

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
