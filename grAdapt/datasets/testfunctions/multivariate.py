# https://en.wikipedia.org/wiki/Test_functions_for_optimization

import numpy as np

# Rastrigin

def rastrigin(x):
    x = np.array(x)
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=0)


def rastrigin_bounds(d):
    """

    Parameters
    ----------
    d : int
        dimension

    Returns
    -------
    list of 2-tuples
    """
    return [(-5.12, 5.12) for _ in range(d)]


def rastrigin_ySol(d):
    return 0


def rastrigin_xSol(d):
    """

    Parameters
    ----------
    d : int
        dimension

    Returns
    -------

    """
    return np.zeros((d,))

# Sphere

def sphere(x):
    x = np.array(x)
    return np.sum(x ** 2)


def sphere_bounds(d):
    """

    Parameters
    ----------
    d : int
        dimension

    Returns
    -------

    """
    return [(-10, 10) for _ in range(d)]


def sphere_ySol(d):
    return 0


def sphere_xSol(d):
    """

    Parameters
    ----------
    d : int
        dimension

    Returns
    -------

    """
    return np.zeros((d,))


# Rosenbrock

def rosenbrock(x):
    x = np.array(x)
    summation = 0
    for i in range(len(x)-1):
        summation += 100*((x[i+1] - x[i]**2)**2) + (1 - x[i])**2

    return summation


def rosenbrock_bounds(d):
    """

    Parameters
    ----------
    d : int
        dimension

    Returns
    -------

    """
    return [(-10, 10) for i in range(d)]


def rosenbrock_ySol(d):
    return 0


def rosenbrock_xSol(d):
    return np.ones((d,))


# Styblinski

def styblinski(x):
    x = np.array(x)
    return np.sum(x ** 4 - 16 * x ** 2 + 5 * x) / 2


def styblinski_bounds(d):
    """

    Parameters
    ----------
    d : int
        dimension

    Returns
    -------

    """
    return [(-5, 5) for _ in range(d)]


def styblinski_ySol(d):
    return -39.16616 * d


def styblinski_xSol(d):
    return np.ones((d,)) * -2.903534
