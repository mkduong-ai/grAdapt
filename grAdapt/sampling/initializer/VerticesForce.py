# python
# import warnings

# Third party imports
import numpy as np

# grAdapt
from .base import Initial
from grAdapt.utils.sampling import sample_corner_bounds


class VerticesForce(Initial):
    """
    Samples all vertices if n_evals >= 2 ** len(bounds).
    Else, a subset of vertices is sampled.
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
        
        if 2 ** len(self.bounds) >= self.n_evals:
            # sample corner points first which fits in n_evals
            d_tilde = int(np.floor(np.log2(self.n_evals)))
            corners_d_tilde = sample_corner_bounds(self.bounds[:d_tilde])  # (2 ** d_tilde, d_tilde) array
            n_tilde = 2 ** d_tilde
            # sample fixed corner points
            fix_corners = np.array([low for (low, high) in self.bounds[d_tilde:]]).reshape(1, -1)
            fix_corners_2d = np.tile(fix_corners, (n_tilde, 1))  # (n, d-d_tilde)

            # corner points with fixed rest dimensions
            corner_points_fixed = np.hstack((corners_d_tilde, fix_corners_2d))

            # because 2 ** n_tilde <= n, sample n - n_tilde
            if self.n_evals - n_tilde > 0:
                random_points = self.sampling_method.sample(bounds=self.bounds,
                                                            n=(self.n_evals - n_tilde),
                                                            x_history=corner_points_fixed)
                return np.vstack((corner_points_fixed, random_points))
            else:
                return corner_points_fixed

        else:
            corner_points = sample_corner_bounds(self.bounds)
            num_corner_points = corner_points.shape[0]
            random_points = self.sampling_method.sample(bounds=self.bounds,
                                                        n=(self.n_evals - num_corner_points),
                                                        x_history=corner_points)

            return np.vstack((corner_points, random_points))
