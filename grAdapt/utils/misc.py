import numpy as np


def random_starts(n_calls, dim):
    """Returns the number of random_starts which should be made
    Parameters
    ----------
    n_calls : number of iterations
    dim : number of dimensions of the function

    Returns
    -------
    numeric

    """
    # Exception handling
    if dim <= 0:
        raise Exception("Dimension dim must be equal or higher than 1.")

    if n_calls <= 0:
        raise Exception("n_evals must be equal or higher than 1.")

    fraction = 0.75 * (1 - (1 / np.log(n_calls + np.e))) * (1 - 1 / np.log(dim + np.e))
    return max(1, int(n_calls * fraction))


def is_inside_relative_range(value, ref_value, pct):
    """

    Parameters
    ----------
    value : numeric
    ref_value : numeric
    pct : numeric
        pct should be smaller than 1.

    Returns
    -------
    boolean
    """
    if ref_value * (1 - pct) <= value <= ref_value * (1 + pct):
        return True
    else:
        return False


def epochs(iteration):
    return min(100, int(iteration ** 1.25))
