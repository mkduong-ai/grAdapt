# Python Standard Libraries
from itertools import product
import warnings

# Third party imports
import numpy as np

from grAdapt.utils.math.spatial import pairwise_distances


def sample_corner_bounds(bounds):
    """Samples points on the corner of given bounds

    Parameters
    ----------
    bounds : list of tuples.
        Each tuple in the list defines the bounds for the corresponding variable
        Example: [(1, 2), (2, 3), (-1, 4)...]

    Returns
    -------
    array (2**len(bounds), len(bounds))

    Examples
    --------
    >>> import grAdapt.utils.sampling as sampling

    Simple example:

    >>> bounds = [(-1, 1), (2, 4)]
    >>> corner_points = sampling.sample_corner_bounds(bounds)
    >>> print(corner_points)
    array([[-1.  2.]
           [-1.  4.]
           [ 1.  2.]
           [ 1.  4.]])

    Using grAdapt Datetype objects:

    >>> from grAdapt.space.datatype import Float, Integer
    >>> range1 = Float(-1.4, 2.3)
    >>> range2 = Integer(2, 3)
    >>> bounds = [range1, range2]
    >>> corner_points = sampling.sample_corner_bounds(bounds)
    >>> print(corner_points)
    array([[-1.4  2. ]
           [-1.4  3. ]
           [ 2.3  2. ]
           [ 2.3  3. ]])
    """
    warnings.warn('Sampling corner points can cause memory leaks. Number of corner points rises exponentially with '
                  'each additional dimension. Use with caution.', ResourceWarning)

    return np.array(list(product(*bounds)), dtype=np.float)


def sample_points_bounds(bounds, n=1, random_state=None):
    """Random sample points uniformly given bounds

    Parameters
    ----------
    bounds : list of tuples.
        Each tuple in the list defines the bounds for the corresponding variable
        Example: [(1, 2), (2, 3), (-1, 4)...]
    n : integer
        Describes the number of points which should be sampled from the given
        bound.
    random_state : int
        random seed

    Returns
    -------
    sampled_points : array-like (n, dim)
        Returns a 2D array. dim is the dimension of a single point
        Each row corresponds to a single point.
        Each column corresponds to a dimension.
    """
    # setting the seed
    if random_state is not None:
        np.random.seed(random_state)

    if n <= 0:
        raise Exception("n should be higher than 0.")
    
    dim = len(bounds)
    sampled_points = np.empty((n, dim))
    for j in range(dim):
        sampled_points[:, j] = np.random.uniform(bounds[j][0], bounds[j][1], (n,))

    return sampled_points


# @jit(nopython=True)
def inside_bounds(bounds, x_next):
    """Checks whether the point x_next is inside of bounds

    Parameters
    ----------
    bounds : list of tuples
        Each tuple in the list defines the bounds for the corresponding variable
        Example: [(1, 2), (2, 3), (-1, 4)...]
    x_next : array_like (d, )


    Returns
    -------
    boolean

    """
    n = len(bounds)
    for i in range(n):
        if not (bounds[i][0] <= x_next[i] <= bounds[i][1]):
            return False

    return True


def inside_bounds_ndim(bounds, x):
    """Checks whether all n samples in x lie inside of bounds

    Parameters
    ----------
    bounds : list of tuples
        Each tuple in the list defines the bounds for the corresponding variable
        Example: [(1, 2), (2, 3), (-1, 4)...]
    x : array_like (n, d)
        n samples with d dimensions


    Returns
    -------
    boolean

    """
    return all(list(map(lambda x: inside_bounds(bounds, x), x)))


def bounds_range(bounds_tuple):
    """Returns the range of a single bound
    Parameters
    ----------
    bounds_tuple : tuple object

    Returns
    -------
    numeric
    """
    return np.abs(bounds_tuple[1] - bounds_tuple[0])


def bounds_range_ndim(bounds):
    """Returns the range of a single bound
    Parameters
    ----------
    bounds : list of tuples
        Each tuple in the list defines the bounds for the corresponding variable
        Example: [(1, 2), (2, 3), (-1, 4)...]

    Returns
    -------
    array-like of shape (len(bounds),)
    """
    return np.array(list(map(bounds_range, bounds)))