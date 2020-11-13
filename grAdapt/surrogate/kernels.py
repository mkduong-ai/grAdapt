# Third party imports
import numpy as np
from sklearn.gaussian_process.kernels import RBF as skRBF, RationalQuadratic as skRationalQuadratic

# grAdapt
from grAdapt.utils.math.linalg import inv_stable


class Nystroem:
    """Formula derived by William and Seeger
    Using the Nyström Method to Speed Up Kernel Machines (2001)

    Nystroem Method: avoid calculating all kernel entries
    Kernel is passed uninitialized.

    Nystroem can be used to approximate the Gram Matrix:
    >>> from grAdapt.surrogate.kernels import Nystroem, RationalQuadratic
    >>> import numpy as np
    >>> xn = np.linspace(-10, 10)
    >>> kernel = RationalQuadratic()
    >>> gram_exact = kernel(xn)
    >>> kernel_nystroem = Nystroem(kernel)
    >>> gram_approx = kernel_nystroem(xn)
    >>> print(np.linalg.norm(gram_exact-gram_approx))

    Nystroem kernel can be passed to the surrogate just like any kernel:
    >>> from grAdapt.surrogate.kernels import Nystroem, RBF
    >>> from grAdapt.surrogate import GPRSlidingWindow
    >>> from grAdapt.models import Sequential
    >>> rbf = RBF()
    >>> kernel = Nystroem(rbf)
    >>> gpr = GPRSlidingWindow(kernel=Nystroem, window_size = 10)
    >>> model = Sequential(surrogate=gpr)
    """

    def __init__(self, kernel, n_components=100):
        """

        Parameters
        ----------
        kernel : kernel initialized object
        n_components : int
            number of rows to be evaluated
        # kernel_params : kernel parameters e.g. length_scale, alpha etc
        """
        self.kernel = kernel
        self.n_components = n_components
        self.kernel_name = self.kernel.__str__()

        if hasattr(self.kernel, 'alpha'):
            self.alpha = self.kernel.alpha
        if hasattr(self.kernel, 'length_scale'):
            self.length_scale = self.kernel.length_scale

    def __call__(self, X1, X2=None):
        """

        Parameters
        ----------
        X1 : array-like (n, d)
        X2 : array-like (n, d)

        Returns
        -------
        array-like (n, n)
            Kernel Matrix
        """

        if len(X1.shape) == 1:
            X1 = X1.reshape(1, -1)

        if X2 is None:
            X2 = X1

        n = X1.shape[0]
        k = self.n_components

        if k < n:
            AC = self.kernel(X1[:k], X2)
            A = AC[:, :k]
            C = AC[:, k:].T

            # Cholesky inverse for numerical stability
            B = C @ inv_stable(A) @ C.T
            CB = np.hstack((C, B))

            del B
            del C
            del A

            return np.vstack((AC, CB))
        else:
            return self.kernel(X1, X2)

    def get_params(self, deep=True):
        return self.kernel.get_params(deep)


def set_lengthscale(bounds):
    """

    Parameters
    ----------
    bounds : list of tuples or list of grAdapt.space.datatype.base
            Each tuple in the list defines the bounds for the corresponding variable
            Example: [(1, 2), (2, 3), (-1, 4)...]

    Returns
    -------

    """
    return np.sqrt(np.sum([(np.abs(u-l)/3)**2 for (l, u) in bounds]))


class RBF(skRBF):
    """Radial Basis Function
    Modified length scale to counter curse of dimensionality.
    """
    def __init__(self, bounds=None):
        super().__init__()
        if bounds is not None:
            self.length_scale = set_lengthscale(bounds)

        self.__name__ = 'RBF'

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
         @ docstring by scikit-learn
         https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/gaussian_process/kernels.py#L1315

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        if self.length_scale == 1.0:
            self.length_scale = np.sqrt(X.shape[1])

        return super().__call__(X, Y, eval_gradient)

    def derivative(self):
        pass


class RationalQuadratic(skRationalQuadratic):
    """Rational Quadratic function
     Modified length scale to counter curse of dimensionality.
    """
    def __init__(self, bounds=None):
        super().__init__()
        if bounds is not None:
            self.length_scale = set_lengthscale(bounds)

        self.__name__ = 'RationalQuadratic'

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
         @ docstring by scikit-learn
         https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/gaussian_process/kernels.py#L1315

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        if self.length_scale == 1.0:
            self.length_scale = np.sqrt(X.shape[1])

        return super().__call__(X, Y, eval_gradient)

    def derivative(self):
        pass