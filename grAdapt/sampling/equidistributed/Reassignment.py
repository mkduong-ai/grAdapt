# Python Standard Libraries
import numpy as np

# grAdapt
from .base import Equidistributed
from grAdapt.utils.sampling import sample_points_bounds
from grAdapt.utils.math.spatial import pairwise_distances


class Reassignment(Equidistributed):
    """Reassignment sampling method (slowest method)

    n Points are initially uniformly sampled given the bounds. The point
    with the smallest distance to another is being resampled uniformly.
    Avoids 'evil neighbours' but is slowest.
    For a better performance: Sliding window has been added.

    """

    def __init__(self, n_iters=None, window_size=500):
        """
        Parameters
        ----------
        n_iters : integer
            number of iterations to reassign worse points
        window_size : integer
            size of history points to consider. smaller is faster but worsen the results.
        """
        super().__init__()
        self.n_iters = n_iters
        self.window_size = window_size

    def sample(self, bounds, n, x_history=None):
        """Samples low discrepancy/equidistributed sequences
        Method has to handle with new bounds and n.

        Parameters
        ----------
        bounds : list of tuples or list of grAdapt.space.datatype.base
            Each tuple in the list defines the bounds for the corresponding variable
            Example: [(1, 2), (2, 3), (-1, 4)...]
        n : int
            number of points to be sampled
        x_history : array-like (2d)
            History points. Consider those to prevent sampling in dense regions.

        Returns
        -------
        array-like (n, len(bounds))
            Returns a 2D array. dim is the dimension of a single point
            Each row corresponds to a single point.
            Each column corresponds to a dimension.
        """

        # set to new variables
        super().sample(bounds, n, x_history)

        if self.n_iters is None:
            self.n_iters = min(int(self.n//2), 1)

        if x_history is None:
            x_history = sample_points_bounds(self.bounds, 1).reshape(1, -1)

        p = sample_points_bounds(self.bounds, self.n)
        p_x_history = np.vstack((x_history, p))
        for i in range(self.n_iters):
            dists_matrix = pairwise_distances(p_x_history[-self.window_size:], p_x_history[-self.window_size:])
            np.fill_diagonal(dists_matrix, np.inf)
            idx_min_eps = np.argmin(np.min(dists_matrix, axis=1))
            p_x_history[-self.window_size:][idx_min_eps] = sample_points_bounds(self.bounds, 1)

        return p_x_history[-self.n:]
