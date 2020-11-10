# decorators
from abc import abstractmethod


class Equidistributed:
    """
    Equidistributed Base class
    It samples points with sample() by given bounds and n
    """

    def __init__(self):
        """
        Parameters
        ----------

        Notes
        -----
        All subclasses can be initialized without parsing arguments.
        """
        self.bounds = None
        self.n = None
        self.window_size = 0

    @abstractmethod
    def sample(self, bounds, n, x_history=None):
        """Samples low discrepancy/equidistributed sequences
        Method has to handle with new bounds and n_evals.

        Parameters
        ----------
        bounds : list of tuples or list of grAdapt.space.datatype.base
            Each tuple in the list defines the bounds for the corresponding variable
            Example: [(1, 2), (2, 3), (-1, 4)...]
        n : int
            number of points to be sampled
        x_history : 2d array-like
            History points. Consider those to prevent sampling in dense regions.

        Returns
        -------
        (n, len(bounds)) numpy array
        """
        self.bounds = bounds
        self.n = n

        # raise Exceptions
        if self.n == 0:
            raise ValueError('n is zero. You must sample with a positive n.')
