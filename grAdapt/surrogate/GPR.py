# Python Standard Libraries
# decorators
from abc import abstractmethod
from deprecated import deprecated
import warnings

# ignore scipy warnings for GPR
warnings.filterwarnings('ignore')

# Third party imports
import numpy as np
import scipy.linalg
from sklearn.gaussian_process import GaussianProcessRegressor

# Package imports
from .base import Surrogate
from .kernels import Nystroem, RationalQuadratic  # , RBF

# grAdapt
from grAdapt.utils.math.linalg import inv_stable


class GPR(Surrogate):
    # TODO: This class does not work with Nystroem Kernel
    """Gaussian Process Regression surrogate
    Based on scikit-learn GaussProcessRegression.
    Implements various gradient calculations depending
    on the chosen kernel.
    """

    def __call__(self):
        pass

    def __init__(self, **surrogate_params):
        """
        Parameters
        ----------
        surrogate_params : dictionary
        """
        if surrogate_params is None or len(surrogate_params) == 0:
            surrogate_params = {'kernel': RationalQuadratic()}
            # surrogate_params = {'kernel': Nystroem(RationalQuadratic)}
        super().__init__(GaussianProcessRegressor(**surrogate_params), surrogate_params)
        self.kernel_function = surrogate_params['kernel']
        self.alpha = None

    @abstractmethod
    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        self.model.fit(X, y)
        self.alpha = self.model.alpha_

    @abstractmethod
    def predict(self, x_predict, **args):
        if len(x_predict.shape) == 1:
            x_predict = x_predict.reshape(1, -1)
        return self.model.predict(x_predict, **args)

    @deprecated("This function is deprecated. The gradient evaluation is not vectorized.")
    def eval_gradient_old(self, x, surrogate_grad_params):
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
        kernel_matrix = self.kernel_function(x, x_train)
        length_scale = self.kernel_function.length_scale
        n, m = kernel_matrix.shape[0], kernel_matrix.shape[1]
        d = x.shape[1]
        grad = np.zeros(d)

        # calculate gradient
        for k in range(d):
            tmp_kernel = np.copy(kernel_matrix)
            for j in range(m):
                tmp_kernel[0, j] = ((x_train[j, k] - x[0, k]) * tmp_kernel[0, j]) / (length_scale ** 2)

            grad[k] = tmp_kernel.dot(self.alpha)

        return grad

    def grad_RBF(self, x, surrogate_grad_params):
        x_train = surrogate_grad_params[0]
        x = x.reshape(1, -1)
        kernel_matrix = self.kernel_function(x, x_train)
        length_scale = self.kernel_function.length_scale
        n, m = kernel_matrix.shape[0], kernel_matrix.shape[1]
        d = x.shape[1]
        grad = np.zeros(d)

        # calculate gradient
        for k in range(d):
            tmp_kernel = np.copy(kernel_matrix)
            tmp_kernel[0, :] = ((x_train[:, k] - x[0, k]) * tmp_kernel[0, :]) / (length_scale ** 2)

            grad[k] = tmp_kernel.dot(self.alpha)

        return grad

    def grad_RationalQuadratic(self, x, surrogate_grad_params):
        x_train = surrogate_grad_params[0]
        x = x.reshape(1, -1)
        kernel_matrix = self.kernel_function(x, x_train)
        length_scale = self.kernel_function.length_scale
        alpha = self.kernel_function.alpha  # this alpha here is a parameter for the kernel!
        n, m = kernel_matrix.shape[0], kernel_matrix.shape[1]
        d = x.shape[1]
        grad = np.zeros(d)

        # calculate gradient
        for k in range(d):
            tmp_kernel = np.copy(kernel_matrix)
            d2 = (np.sum((x[0, k] - x_train[:, k]) ** 2)) / (2 * alpha * length_scale ** 2)
            tmp_kernel[0, :] = -(2 * x[0, k] - 2 * x_train[:, k]) / (2 * length_scale ** 2) * (
                    tmp_kernel[0, :] / (1 + d2))

            grad[k] = tmp_kernel.dot(self.alpha)

        return grad

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

        kernel_name = self.kernel_function.__str__()
        if 'Nystroem' in self.kernel_function.__str__():
            kernel_name = self.kernel_function.kernel_name

        if 'RationalQuadratic' in kernel_name:
            return self.grad_RationalQuadratic(x, surrogate_grad_params)
        elif any(list(map(lambda x: x in kernel_name, ['None', 'RBF']))):
            return self.grad_RBF(x, surrogate_grad_params)
        else:
            raise AttributeError('Given kernel function of surrogate model (GPR) is not supported. '
                                 'Set a different Kernel. '
                                 'Used Kernel: ' + kernel_name())


class GPROnlineLearningWrapper(GPR):
    """Gaussian Process Regression (Subclass of GPR)
    Do not use this class!
    Superclass of many Online Learning Implementations
    for GPR.

    """

    def __init__(self, **surrogate_params):
        """
        Parameters
        ----------
        surrogate_params : dictionary
        """
        if surrogate_params is None or len(surrogate_params) == 0:
            surrogate_params = {'kernel': RationalQuadratic()}
            #surrogate_params = {'kernel': Nystroem(RationalQuadratic)}
        self.surrogate_params = surrogate_params
        self.kernel_function = surrogate_params['kernel']
        self.kernel_matrix = None
        self.kernel_matrix_inv = None
        self.alpha = None
        self.X_train = None

    def insert(self, X1, X2=None):
        """Calculates Covariance/Kernel matrix and the inverse of it
        Instead of recalculating, Kernel Matrix is updated
        by stacking a row and a column on the right

        Parameters
        ----------
        X1 : array-like 2D
        X2 : array-like 2D

        Returns
        -------

        """
        if X2 is None:
            X2 = X1

        # Training continuation
        if self.kernel_matrix is None:
            self.kernel_matrix = self.kernel_function(X1, X2)
            self.kernel_matrix_inv = inv_stable(self.kernel_matrix)
        else:
            b = self.kernel_function(X1[-1].reshape(1, -1), X2)
            self.kernel_matrix = np.concatenate((self.kernel_matrix, b[0, :-1].reshape(1, -1)), axis=0)
            self.kernel_matrix = np.concatenate((self.kernel_matrix, b.T), axis=1)

            # Inversion of kernel matrix
            # Augmented Inverse Update
            d = b[0, -1].reshape(1, 1)
            b = b[0, :-1].reshape(-1, 1)
            e = self.kernel_matrix_inv @ b
            g = 1 / (d - b.T @ e)

            self.kernel_matrix_inv = self.kernel_matrix_inv + g * e @ e.T
            row_add = (-g * e).T
            self.kernel_matrix_inv = np.concatenate((self.kernel_matrix_inv, row_add.reshape(1, -1)),
                                                    axis=0)
            column_add = np.concatenate((row_add, g), axis=1)
            self.kernel_matrix_inv = np.concatenate((self.kernel_matrix_inv, column_add.T), axis=1)

    def delete(self, X1, X2=None):
        """Calculates Covariance/Kernel matrix and the inverse of it
        Instead of recalculating, Kernel Matrix is updated

        Parameters
        ----------
        X1 : array-like 2D
        X2 : array-like 2D

        Returns
        -------

        """
        if X2 is None:
            X2 = X1

        # Training continuation
        if self.kernel_matrix is None:
            self.kernel_matrix = self.kernel_function(X1, X2)
            self.kernel_matrix_inv = inv_stable(self.kernel_matrix)
        else:
            # delete first row and column
            self.kernel_matrix = self.kernel_matrix[1:, 1:]

            # insert new data point
            h = self.kernel_function(X1[-1].reshape(1, -1), X2[-self.window_size:, :]).T
            g = h[-1, 0]
            self.kernel_matrix = np.concatenate((self.kernel_matrix, h[:-1, 0].reshape(-1, 1).T), axis=0)
            self.kernel_matrix = np.concatenate((self.kernel_matrix, h), axis=1)

            # Inversion of kernel matrix
            # Augmented Inverse Update
            e = self.kernel_matrix_inv[0, 0]
            f = self.kernel_matrix_inv[1:, 0]
            H = self.kernel_matrix_inv[1:, 1:]

            B = H - (f @ f.T) / e
            h = h[:-1, :]
            s = 1 / (g - h.T @ B @ h)
            self.kernel_matrix_inv = B + (B @ h) @ (B @ h).T * s

            # add rows and columns
            row_add = - (B @ h).T * s
            column_add = np.concatenate((row_add, s), axis=1)
            self.kernel_matrix_inv = np.concatenate((self.kernel_matrix_inv, row_add), axis=0)
            self.kernel_matrix_inv = np.concatenate((self.kernel_matrix_inv, column_add.T), axis=1)

    @abstractmethod
    def fit(self, X, y):
        pass

    def predict(self, x_predict, return_std=False):
        if self.kernel_matrix is None:
            raise Exception('Make sure to fit model before using predict.')

        if len(x_predict.shape) == 1:
            x_predict = x_predict.reshape(1, -1)

        # positive definite matrix
        Id = np.eye(self.kernel_matrix.shape[0])
        self.kernel_matrix = self.kernel_matrix + Id * 1e-7

        pred_matrix = self.kernel_function(x_predict, self.X_train)

        if return_std:
            sigma_22 = self.kernel_function(x_predict)
            solved = scipy.linalg.solve(self.kernel_matrix, pred_matrix.T, assume_a='pos').T
            posterior_cov = sigma_22 - solved @ pred_matrix.T
            sigma = np.sqrt(np.maximum(0, np.diag(posterior_cov)))
            return pred_matrix @ self.alpha, sigma
        else:
            return pred_matrix @ self.alpha

    def negative_marginal_likelihood(self):
        pass


class GPROnlineInsert(GPROnlineLearningWrapper):
    """Gaussian Process Regression (Subclass of GPROnlineLearningWrapper)
    GPR Online Learning with Augmented Inverse Update
    @ Formulas derived by https://github.com/Bigpig4396
    @ Original paper: Incremental Sparsification for Real-time Online Model Learning (Nguyen, Peters)

    In order to fit a new point, a history of points is required i.e.:
    >>> import numpy as np
    >>> # Initialize Training Data
    >>> x = np.linspace(-10, 10).reshape(-1, 1)
    >>> y = np.random.random(x.shape[0])
    >>> # Fit with GPROnlineInsert
    >>> gpr = GPROnlineInsert()
    >>> for i in range(x.shape[0]):
    >>>     gpr.fit(x[:i], y[:i])
    """

    def __init__(self, **surrogate_params):
        """
        Parameters
        ----------
        surrogate_params : dictionary
        """
        super().__init__(**surrogate_params)

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        self.X_train = X

        if self.kernel_matrix is not None:
            for i in range(self.kernel_matrix.shape[0], self.X_train.shape[0]):
                self.insert(X[:i + 1, :])
        else:
            self.insert(X)
        self.alpha = self.kernel_matrix_inv @ y


class GPRSlidingWindow(GPROnlineLearningWrapper):
    """Gaussian Process Regression (Subclass of GPROnlineLearningWrapper)
    GPR Online Learning with Sliding Window

    @ Formulas derived by https://github.com/Bigpig4396
    @ Original papers :
                        Incremental Sparsification for Real-time Online Model Learning (Nguyen, Peters, 2011)
                        Kernel Adaptive Learning (Principe, Liu, 2008)

    Only keeps the latest window_size points where window_size is a user-defined parameter.

    In order to fit a new point, a history of points is required i.e.:
    >>> import numpy as np
    >>> # Initialize Training Data
    >>> x = np.linspace(-10, 10, 1000).reshape(-1, 1)
    >>> y = np.random.random(x.shape[0])
    >>> # Fit with GPROnlineInsert
    >>> gpr = GPRSlidingWindow(window_size=100)
    >>> for i in range(x.shape[0]):
    >>>     gpr.fit(x[:i], y[:i])
    """

    def __init__(self, **surrogate_params):
        """
        Parameters
        ----------
        surrogate_params : dictionary
        """
        super().__init__(**surrogate_params)
        if 'window_size' not in self.surrogate_params:
            self.surrogate_params['window_size'] = 300
        self.window_size = self.surrogate_params['window_size']

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        self.X_train = X

        if self.X_train.shape[0] > self.window_size:
            X = X[-self.window_size:, :]
            self.delete(X, X)
        else:
            # insert with training continuation
            if self.kernel_matrix is not None:
                for i in range(self.kernel_matrix.shape[0], self.X_train.shape[0]):
                    self.insert(X[:i + 1, :], X[:i + 1, :])
            else:
                self.insert(X, X)

        self.alpha = self.kernel_matrix_inv @ y[-self.window_size:]

    def predict(self, x_predict, return_std=False):
        self.X_train = self.X_train[-self.window_size:, :]
        return super().predict(x_predict, return_std)

    def grad_RBF(self, x, surrogate_grad_params):
        surrogate_grad_params_window = surrogate_grad_params.copy()
        x_train = surrogate_grad_params_window[0]
        if x_train.shape[0] > self.window_size:
            x_train = x_train[-self.window_size:, :]

        surrogate_grad_params_window[0] = x_train
        return super().grad_RBF(x, surrogate_grad_params_window)

    def grad_RationalQuadratic(self, x, surrogate_grad_params):
        surrogate_grad_params_window = surrogate_grad_params.copy()
        x_train = surrogate_grad_params_window[0]
        if x_train.shape[0] > self.window_size:
            x_train = x_train[-self.window_size:, :]

        surrogate_grad_params_window[0] = x_train
        return super().grad_RationalQuadratic(x, surrogate_grad_params_window)


class GPROnlineStandard(GPROnlineLearningWrapper):
    """Gaussian Process Regression surrogate
    GPR Online Learning with Augmented Inverse Update
    @ Formulas derived by https://github.com/Bigpig4396
    @ Original paper: Incremental Sparsification for Real-time Online Model Learning (Nguyen, Peters)

    This class is not meant to be used with Sequential. Use GPROnlineInsert instead.
    It is meant to be used outside of grAdapt.

    Fit single points, no history of data points needed i.e.:
    >>> gpr = GPROnlineStandard()
    >>> gpr.fit(x[:x.shape[0]-1], y[:x.shape[0]-1])
    >>> gpr.fit_online(x[x.shape[0]], y[x.shape[0]])
    """

    def __init__(self, **surrogate_params):
        """
        Parameters
        ----------
        surrogate_params : dictionary
        """
        super().__init__(**surrogate_params)
        self.y_train = None

    def kernel(self, X1, X2=None):
        """Calculates Covariance/Kernel matrix and the inverse of it
        Instead of recalculating, Kernel Matrix is updated
        by stacking a row and a column on the right
        Does not need a history of training data!

        Parameters
        ----------
        X1 : array-like 2D
        X2 : array-like 2D

        Returns
        -------

        """
        if X2 is None:
            X2 = self.X_train

        b = self.kernel_function(X1, X2)
        self.kernel_matrix = np.concatenate((self.kernel_matrix, b[0, :-1].reshape(1, -1)), axis=0)
        self.kernel_matrix = np.concatenate((self.kernel_matrix, b.T), axis=1)

        # Inversion of kernel matrix
        # Augmented Inverse Update
        d = b[0, -1].reshape(1, 1)
        b = b[0, :-1].reshape(-1, 1)
        e = self.kernel_matrix_inv @ b
        g = 1 / (d - b.T @ e)

        self.kernel_matrix_inv = self.kernel_matrix_inv + g * e @ e.T
        row_add = (-g * e).T
        self.kernel_matrix_inv = np.concatenate((self.kernel_matrix_inv, row_add.reshape(1, -1)),
                                                axis=0)
        column_add = np.concatenate((row_add, g), axis=1)
        self.kernel_matrix_inv = np.concatenate((self.kernel_matrix_inv, column_add.T), axis=1)

    def fit(self, X, y):
        """First fit or refit points
        This method does not do online/incremental learning!

        Parameters
        ----------
        X : array-like 2D (n, d)
        y : array-like 1d (n, )

        Returns
        -------

        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        self.X_train = X
        self.y_train = y
        self.kernel_matrix = self.kernel_function(X, X)
        self.kernel_matrix_inv = inv_stable(self.kernel_matrix)
        self.alpha = self.kernel_matrix_inv @ self.y_train

    # TODO: Bug here?
    def fit_online(self, X, y):
        """Fits to GPR online/incremental
        Can only fit a single point.

        Parameters
        ----------
        X : array-like 2D (n, d)
        y : array-like 1d (n, )

        Returns
        -------

        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        self.y_train = np.concatenate((self.y_train, y), axis=0)

        if self.X_train is None:
            self.fit(X, y)
        else:
            initial_size = self.X_train.shape[0]
            self.X_train = np.concatenate((self.X_train, X), axis=0)
            for i in range(X.shape[0]):
                self.kernel(X[i].reshape(1, -1), self.X_train[: i + initial_size + 1])

            self.alpha = self.kernel_matrix_inv @ self.y_train
