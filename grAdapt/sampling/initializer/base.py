# decorators
from abc import abstractmethod


class Initial:
    """
    Initial Base class
    """

    def __init__(self, sampling_method=None):
        """
        Parameters
        ----------
        sampling_method : grAdapt.sampling.equidistributed Object
            Sample low discrepancy sequences when initial point method is not feasible

        """
        self.bounds = None
        self.n_evals = None
        self.sampling_method = sampling_method

    @abstractmethod
    def sample(self, bounds, n_evals):
        """Returns a numpy array of sampled points.
        Might include corners of the hypercube depending on the method.

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
        self.bounds = bounds
        self.n_evals = n_evals
