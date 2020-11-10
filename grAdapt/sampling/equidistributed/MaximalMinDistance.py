# Python Standard Libraries
import numpy as np

# grAdapt
from .base import Equidistributed
from grAdapt.utils.sampling import sample_points_bounds
from grAdapt.utils.math.spatial import pairwise_distances


class MaximalMinDistance(Equidistributed):
    """Maximal min distance sampling method

    A fixed amount of points are candidates. The candidate is chosen
    as a point if it has the highest minimal epsilon margin among
    other points. The minimal epsilon margin is the smallest distance
    between two points. Each point has an minimal epsilon margin.
    For a better performance, only a fixed amount of latest points
    are considered.
    Has a disadvantage: creates 'evil' neighbours.

    """

    def __init__(self, n_candidates=10, window_size=500):
        """
        Parameters
        ----------
        n_candidates : integer
            number of candidates
        window_size : integer
            size of history points to consider. smaller is faster but worsen the results.
        """
        super().__init__()
        self.n_candidates = n_candidates
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

        if x_history is None:
            x_history = sample_points_bounds(self.bounds, 1)
            x_history_list = list(x_history)
        else:
            x_history_list = list(x_history)

        for i in range(self.n):
            x_history_sublist = x_history_list[-self.window_size:]
            candidates = sample_points_bounds(self.bounds, self.n_candidates)
            dists_matrix = pairwise_distances(candidates, np.array(x_history_sublist))
            min_dists = np.min(dists_matrix, axis=1)
            max_min_dists = np.argmax(min_dists)
            x_history_list.append(candidates[max_min_dists])

        return np.array(x_history_list)[-self.n:]
