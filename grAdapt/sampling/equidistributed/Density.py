# Python Standard Libraries
import numpy as np

# grAdapt
from .base import Equidistributed
from grAdapt.utils.sampling import sample_points_bounds, bounds_range
from grAdapt.utils.math.spatial import pairwise_distances


class Density(Equidistributed):
    """Density-based sampling

    For each n_candidates candidate, the number of points withing eps is determined.
    The candidate with the lowest number of points within the eps-radius is sampled.
    Repeated for n.

    Eps is determined by 1/25 * (largest bound).
    Empirically satisfying results with such setting.

    Disadvantages: Creates holes, not reliable in the beginning.
    """

    def __init__(self, n_candidates=5, eps='auto', window_size=500):
        """
        Parameters
        ----------
        n_candidates : integer
            number of candidates
        eps : int or string
            if set to 'auto', then 1/25 of the largest bound is set as eps
            else eps is set as given
        window_size : integer
            size of history points to consider. smaller is faster but worsen the results.
        """
        super().__init__()
        self.n_candidates = n_candidates
        self.eps = eps
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
            x_history_list = list(sample_points_bounds(self.bounds, 1))
        else:
            x_history_list = list(x_history)

        if self.eps == 'auto':
            self.eps = 1 / 25 * np.max(np.array(list(map(bounds_range, self.bounds))))

        start_eps = self.eps

        for i in range(self.n):

            # auto decrease
            self.eps = start_eps/np.log(np.e+i)

            candidates = sample_points_bounds(self.bounds, self.n_candidates)
            dists_matrix = pairwise_distances(candidates, np.array(x_history_list[-self.window_size:]))
            clusters = dists_matrix < self.eps  # cluster for every point in p
            clusters_size = np.sum(clusters, axis=1)  # calc size for each cluster
            # first point tends to be empty already (sparsity at the beginning)
            if clusters_size[0] == 0:
                x_history_list.append(candidates[0])
            else:
                smallest_cluster_idx = np.argmin(clusters_size)
                x_history_list.append(candidates[smallest_cluster_idx])  # add to current points

        return np.array(x_history_list)[-self.n:]
