import scipy.linalg
import numpy as np


def inv_stable(A):
    """Inverts a quadratic matrix with non-negative eigenvalues
    Inverts using cholesky decomposition if matrix is positive semi-definite.
    If not, inverts matrix using eigenvalue decomposition.
    If eigendecomposition does not succeed then use numpy.linalg.inv

    Parameters
    ----------
    A : array-like (n, n)
        Matrix A has non-negative eigenvalues

    Returns
    -------
    array-like (n, n)
    """
    try:
        return inv_chol(A, symmetric=True)
    except:
        try:
            return inv_eig(A)
        except:
            return np.linalg.inv(A)


def inv_eig(A):
    """Inverts a quadratic matrix with non-negative eigenvalues
    Uses eigendecomposition of a matrix.

    Parameters
    ----------
    A : array-like (n, n)

    Returns
    -------
    The inverse array-like (n, n)

    Notes
    -----
    :math:`A = Q V Q^{-1}`
    :math: `A^_{-1} = (Q V Q^{-1})^{-1} = Q^{-1} V^{-1} Q`
    """
    try:
        V, Q = np.linalg.eig(A)
        V = np.diag(1/V)

        # test symmetry
        symmetric = np.allclose(A, A.T, rtol=1e-3, atol=1e-4)

        if symmetric:
            return Q @ V @ Q.T
        else:
            return Q @ V @ np.linalg.inv(Q)
    except:
        raise Exception('Finding inverse of matrix with eigendecomposition failed.')


def inv_chol(A, symmetric=None):
    """Inverts a positive semi-definite matrix A (stable)
    Adds a small positive definite matrix and uses cholesky decomposition.

    Parameters
    ----------
    A : positive semi-definite matrix (n, n)
    symmetric : boolean
        set True if matrix is symmetric. Else false.

    Returns
    -------
    An approximate inverse positive definite matrix (n, n)
    """
    not_2d = len(A.shape) != 2
    try:
        not_quadratic = A.shape[0] != A.shape[1]
    except:
        not_quadratic = True

    # test symmetry
    if symmetric is None:
        symmetric = np.allclose(A, A.T, rtol=1e-3, atol=1e-4)

    if not_2d or not_quadratic:
        raise Exception('Given Matrix is either not 2D or quadratic.')
    elif not symmetric:
        raise Exception('Given matrix is not symmetric.')
    else:
        # make A positive definite for the triangulation
        Id = np.eye(A.shape[0])
        P = Id * 1e-7
        A_tilde = A + P
        # triangulate matrix with cholesky
        L = scipy.linalg.cholesky(A_tilde, lower=True)
        L_inv = scipy.linalg.solve_triangular(L, Id, lower=True)
        A_tilde_inv = L_inv.T @ L_inv

        return A_tilde_inv
