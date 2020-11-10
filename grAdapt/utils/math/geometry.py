import numpy as np
from scipy.special import gamma

# package
from grAdapt.utils.sampling import bounds_range


def volume_hypersphere(dim, radius=1):
    """Calculates the volume of a hypersphere given the radius and the dimension

    Parameters
    ----------
    dim : int
    radius : float

    Returns
    -------
    float
    """
    return (radius ** dim) * np.pi**(dim/2) / (gamma(dim/2 + 1))


def volume_hyperrectangle(bounds):
    """Calculates the volume of a hyperrectangle of any dimension

    Parameters
    ----------
    bounds : list of tuples.
        Each tuple in the list defines the bounds for the corresponding variable
        Example: [(1, 2), (2, 3), (-1, 4)...]

    Returns
    -------

    """
    return np.prod(list(map(bounds_range, bounds)))
