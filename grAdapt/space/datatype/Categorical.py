# Python Standard Libraries
import numpy as np

# grAdapt
from .base import Datatype


class Categorical(Datatype):
    """Interally uses Ordinal Datatype
    Can be used for ordinal categories. A list of strings is used as input. The order is given by the list.
    An example to define age categories:

    >>> from grAdapt.space.datatype import Categorical
    >>> age_category = Categorical(['young', 'middleaged', 'old'])
    >>> bounds = [age_category]

    Internally, each category is treated as an Integer.
    """

    def __init__(self, list_str):
        """

        Parameters
        ----------
        list_str : list of strings
            Each string represents a category
        """

        self.len = len(list_str)
        self.bound = [0, self.len-1]
        self.categories = list_str
        self.dtype = 'categorical'

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
        return 'Categorical('+str(self.categories)+')'

    def __str__(self):
        return 'Categorical('+str(self.categories)+')'

    def transform(self, x):
        """

        Parameters
        ----------
        x : numeric

        Returns
        -------
        numeric

        """
        return self.get_value(x)
