# Python Standard Libraries
# decorators
import warnings
from deprecated import deprecated

# Third party imports
import numpy as np

# Package imports
from .base import Surrogate


class NoModel(Surrogate):
    """NoModel Surrogate
    No surrogate model is used here
    The gradients are approximated by partial difference quotient
    @ https://en.wikipedia.org/wiki/Partial_derivative
    Number of function evaluations is n_evals * dim * 100

    h : Parameter based on search space
    """
    def __init__(self, **surrogate_params):
        super().__init__(None, surrogate_params)
        self.X_train = None
        warnings.warn('You are using no surrogate model. The number of function'
                      ' evaluations is therefore much higher than expected. However, it'
                      ' can perform better on higher dimensions.')

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

        if len(X.shape) == 1:
            raise Exception('If \'NoModel\' is used, please enter a higher function'
                            ' evaluation number \'n_evals\'.')
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
        if return_std:
            return np.zeros_like(x_predict), np.zeros_like(x_predict)
        else:
            return np.zeros_like(x_predict)

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
        # x_train = surrogate_grad_params[0]
        # y_train = surrogate_grad_params[1]
        func = surrogate_grad_params[2]
        bounds = surrogate_grad_params[3]

        grad = np.zeros_like(x)
        x_copy = np.copy(x)
        # difference quotient
        for i in range(x.shape[0]):
            hi = np.min(np.array([np.abs(bounds[i][0]-bounds[i][1])/100, 1e-2]))
            original_value = x[i]
            x_tilde = x

            # perturbation of xi with h
            xi_perturbation = x_tilde[i] + hi
            if not(bounds[i][0] <= xi_perturbation <= bounds[i][1]):
                xi_perturbation = x_tilde[i] - hi
                hi = -hi
            x_tilde[i] = xi_perturbation

            # gradient
            if hi != 0:
                dfdxi = (func(x_tilde)-func(x_copy))/hi
            else:
                dfdxi = 0

            grad[i] = dfdxi

            # revert back to original value
            x_tilde[i] = original_value
            x[i] = original_value

        return grad

    @deprecated("This function is deprecated and very slow due to np.copy")
    def eval_gradient_deprecated(self, x, surrogate_grad_params):
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
        # x_train = surrogate_grad_params[0]
        # y_train = surrogate_grad_params[1]
        func = surrogate_grad_params[2]
        bounds = surrogate_grad_params[3]

        grad = np.zeros_like(x)
        # difference quotient
        for i in range(x.shape[0]):
            hi = np.min(np.array([np.abs(bounds[i][0]-bounds[i][1])/100, 1e-2]))
            x_tilde = np.copy(x)

            # perturbation of xi with h
            xi_perturbation = x_tilde[i] + hi
            if not(bounds[i][0] <= xi_perturbation <= bounds[i][1]):
                xi_perturbation = x_tilde[i] - hi
                hi = -hi
            x_tilde[i] = xi_perturbation

            # gradient
            if hi != 0:
                dfdxi = (func(x_tilde)-func(x))/hi
            else:
                dfdxi = 0

            grad[i] = dfdxi

        return grad
