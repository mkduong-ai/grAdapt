# python
import warnings

# grAdapt
from .base import Initial


class Standard(Initial):
    """
    Simply uses a low discrepancy/equidistributed method and samples points by it.
    """
    def __init__(self, sampling_method=None):
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
        return self.sampling_method.sample(bounds=bounds, n=n_evals)
