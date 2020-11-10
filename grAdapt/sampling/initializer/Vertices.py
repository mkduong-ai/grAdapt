# python
# import warnings

# Third party imports
import numpy as np

# grAdapt
from .base import Initial
from grAdapt.utils.sampling import sample_corner_bounds


class Vertices(Initial):
    """
    Samples vertices if n_evals >= 2 ** len(bounds).
    Else low discrepancy sequences are sampled.
    """

    def __init__(self, sampling_method):
        """
        Parameters
        ----------
        sampling_method : grAdapt.sampling.equidistributed Object
            Sample low discrepancy sequences when initial point method is not feasible
        """
        super().__init__(sampling_method)

    def sample(self, bounds, n_evals):
        """Returns a numpy array of sampled points.
        Does not include corner points of the hypercube/search space.

        Parameters
        ----------
        bounds : list of tuples or list of grAdapt.space.datatype.base
            Each tuple in the list defines the bounds for the corresponding variable
            Example: [(1, 2), (2, 3), (-1, 4)...]
        n_evals : int
            number of initial points sampled by method

        Returns
        -------
        (self.n_evals, len(self.bounds)) numpy array
        """
        super().sample(bounds, n_evals)
        if 2 ** len(self.bounds) > self.n_evals:
            return self.sampling_method.sample(bounds=bounds, n=n_evals)
        else:
            corner_points = sample_corner_bounds(self.bounds)
            num_corner_points = corner_points.shape[0]
            if self.n_evals > 2 ** len(self.bounds):
                random_points = self.sampling_method.sample(bounds=self.bounds,
                                                            n=(self.n_evals - num_corner_points),
                                                            x_history=corner_points)
                return np.vstack((corner_points, random_points))
            else:
                return corner_points
