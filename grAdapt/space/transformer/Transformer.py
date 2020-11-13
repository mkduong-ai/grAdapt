# Python Standard Libraries
import numpy as np

# decorators
from abc import abstractmethod

# Third party imports
from scipy.stats import loguniform

# grAdapt


class Transformer:

    def __init__(self, func, bounds, transform='identity'):
        """

        Parameters
        ----------
        func : function to transform the input
        bounds : list of Datatype's
        transform : str
            `identity` : no changes are applied
            `normalize` : normalize each dimension of bounds to [0, 1]
        """
        self.func = func
        self.bounds = bounds

    def __call__(self, x):
        """

        Parameters
        ----------
        x : array-like (d, )

        Returns
        -------

        """
        x_transformed = np.zeros_like(x, dtype='O')

        for i in range(x.shape[0]):
            try:
                x_transformed[i] = self.bounds[i].transform(x[i])
            except:
                x_transformed[i] = x[i]

        if all(isinstance(x, float) for x in x_transformed):
            x_transformed = np.array(x_transformed, dtype=np.float)

        return self.func(x_transformed)
