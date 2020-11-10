# Python Standard Libraries
import numpy as np

# grAdapt
from .base import Datatype


class Ordinal(Datatype):
    """Ordinal Datatype
    Can be used for ordinal categories. A list of strings is used as input. The order is given by the list.
    An example to define age categories:

    >>> from grAdapt.space.datatype import Ordinal
    >>> age_category = Ordinal(['young', 'middleaged', 'old'])
    >>> bounds = [age_category]

    Internally, each category is treated as an Integer.
    """

    def __init__(self, list_str, prior='uniform'):
        """

        Parameters
        ----------
        list_str : list of strings
            Each string represents a category
        prior: str
            - if 'uniform' then nothing is changed
            - if 'log-uniform' then samples between low and high are drawn from the log-uniform distribution
        """

        self.len = len(list_str)
        self.prior = prior
        self.bound = [0, self.len-1]
        self.low = 1
        self.high = self.len
        self.categories = list_str
        self.dtype = 'ordinal'

    def __len__(self):
        return self.len

    def __getitem__(self, key):
        return self.bound[key]

    def __setitem__(self, key, value):
        self.bound[key] = value

    def get_value(self, value):
        return self.get_category(value)

    def get_category(self, value):
        key = int(np.round(value))
        return self.categories[key]

    def __repr__(self):
        return 'Ordinal('+str(self.categories)+')'

    def __str__(self):
        return 'Ordinal('+str(self.categories)+')'

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
            x_normalized = x / (self.high - self.low)
            return self.get_value(self.low * np.power(self.high * 1.0 / self.low, x_normalized))
        else:
            return self.get_value(x)
