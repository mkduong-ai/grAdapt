# Python Standard Libraries
# decorators
from deprecated import deprecated

# Third party imports
import numpy as np

# Package imports
from .base import Surrogate
from .kernels import RBF


class RBFRegression(Surrogate):

    @deprecated("Deprecated object name. The name of this object might be misleading.")
    def __init__(self, **surrogate_params):
        super().__init__(None, surrogate_params)
        self.RBF = RBF()
        self.X_train = None
        self.alpha_ = None

    def fit(self, X, y):
        """Fit training data (X, y)
        Parameters
        ----------
        X : array-like (n, d)
        y : array-like (n,)


        Returns
        -------
        None

        """
        self.alpha_ = y

        if len(X.shape) == 1:
            self.X_train = X.reshape(1, -1)
        else:
            self.X_train = X

    def predict(self, x_predict, return_std=False):
        """Predict y value based on x_predict
        Parameters
        ----------
        x_predict : array-like (n, d)
        return_std : array-like (n,)
            returns the standard deviation of the posterior mean


        Returns
        -------
        array-like (n,)

        """
        if len(x_predict.shape) == 1:
            x_predict = x_predict.reshape(1, -1)
        return self.RBF(x_predict, self.X_train) @ self.alpha_

    def eval_gradient(self, x, surrogate_grad_params):
        """Evaluates the gradient of GPR at point x
        Parameters
        ----------
        x: 1D array-like (d,)
        surrogate_grad_params: list
            surrogate_grad_params[0]: whole training data shape (m, d)


        Returns
        -------
        grad : 1D array-like
            same shape as x
        """

        x_train = surrogate_grad_params[0]
        x = x.reshape(1, -1)
        kernel_matrix = self.RBF(x, x_train)
        length_scale = self.RBF.length_scale
        n, m = kernel_matrix.shape[0], kernel_matrix.shape[1]
        d = x.shape[1]

        grad = np.zeros(d)
        # calculate gradient
        for k in range(d):
            tmp_kernel = np.copy(kernel_matrix)
            tmp_kernel[0, :] = ((x_train[:, k] - x[0, k]) * tmp_kernel[0, :]) / (length_scale ** 2)

            grad[k] = tmp_kernel.dot(self.alpha_)

        return grad

    def kernel_(self, x_test, x_train):
        """Call the kernel function
        Parameters
        ----------
        x_test : array-like (n, d)
        x_train : array-like (m, d)


        Returns
        -------
        array-like (n, m)
        """

        if len(x_test.shape) == 1:
            x_test = x_test.reshape(1, -1)

        if len(x_train.shape) == 1:
            x_train = x_train.reshape(1, -1)

        return self.RBF(x_test, x_train)
