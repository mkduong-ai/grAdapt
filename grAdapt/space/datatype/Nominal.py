# Python Standard Libraries
import numpy as np

# grAdapt
from .base import Datatype


class Nominal(Datatype):
    """Nominal Datatype
    Is used for nominal categories. The categories are not ordered. Example usage:

    >>> from grAdapt.space.datatype import Nominal
    >>> animals = Nominal(['cat', 'dog', '´turtle'])
    >>> bounds = [animals]

    Internally, each category is a dimension of its own. The categories are encoded to one-hot.
    """

    def __init__(self, list_str):
        """

        Parameters
        ----------
        list_str : list of strings
            Each string represents a category
        """
        raise NotImplementedError
        self.len = len(list_str)
        self.bound = [0, self.len-1]
        self.categories = list_str
        self.dtype = 'nominal'

    def __len__(self):
        return self.len

    def __getitem__(self, key):
        return self.bound[key]

    def __setitem__(self, key, value):
        self.bound[key] = value

    def __repr__(self):
        return 'Nominal('+str(self.categories)+')'

    def __str__(self):
        return 'Nominal'+str(self.categories)+')'

    def get_value(self, value):
        # value is onehot encoded
        return self.get_category(value)

    def get_category(self, value):
        # value is onehot encoded
        key = np.argmax(value)
        return self.categories[key]

    def transform(self, x):
        """

        Parameters
        ----------
        x : numeric

        Returns
        -------
        numeric

        """
        closest_category = np.floor(x)
        onehot_encoding = np.zeros((len(self.len),))
        onehot_encoding[closest_category] = 1
        return onehot_encoding
