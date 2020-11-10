# Python Standard Libraries
import numpy as np

# scikit
from sklearn.neighbors import KDTree

# grAdapt
from .base import Equidistributed
from grAdapt.utils.sampling import sample_points_bounds
from grAdapt.utils.math.spatial import pairwise_distances


class KDistanceMean(Equidistributed):
    """K-Distance sampling method

    Candidates are firstly generated. The candidate with the largest k-distance is sampled.
    """

    def __init__(self, n_candidates=10, k=3, window_size=500):
        """
        Parameters
        ----------
        k : integer
            k-th element from a point
        window_size : integer
            size of history points to consider. smaller is faster but worsen the results.
        """
        super().__init__()
        self.n_candidates = n_candidates
        self.k = k
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

            # k-distance
            sorted_dists = np.sort(dists_matrix, axis=1)
            k_dists_mean = np.mean(sorted_dists[:, :self.k], axis=1)
            best_candidate_idx = np.argmax(k_dists_mean)
            x_history_list.append(candidates[best_candidate_idx])

        return np.array(x_history_list)[-self.n:]





