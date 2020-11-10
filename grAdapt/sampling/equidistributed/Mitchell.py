# Python Standard Libraries
import warnings

# grAdapt
# from .base import Equidistributed
from .MaximalMinDistance import MaximalMinDistance


class Mitchell(MaximalMinDistance):
    """
    [Mitchell et al., 1991],
    Spectrally optimal sampling for distribution ray tracing
    """
    def __init__(self, m=3):
        """
        Parameters
        ----------
        m : integer
            number of candidates = m * n
        """
        warnings.warn('Mitchell\' best candidate has a time complexity of O(n^3) '
                      'and memory issues when dealing with higher sample numbers. '
                      'Use MaximalMinDistance instead which is an improved version '
                      'with linear time complexity.', ResourceWarning)
        super().__init__(n_candidates=m, window_size=0)
        self.candidates_set = False

    def sample(self, bounds, n, x_history=None):
        """Samples low discrepancy/equidistributed sequences according to Mitchell.
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
        if self.candidates_set is False:
            self.n_candidates = self.n_candidates * n
            self.candidates_set = True
        return super().sample(bounds, n, x_history)
