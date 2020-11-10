# Python Standard Libraries
from abc import abstractmethod

# Third party imports
import numpy as np

from .base import Optimizer


class RMSProp(Optimizer):
    """Optimizer object

    Implementation of RMSProp

    """
    def __init__(self, surrogate, params=[1e-3, 0.9]):
        super().__init__(surrogate, params)

    def run(self, xp, num_iters, surrogate_grad_params):
        alpha = self.params[0] # learning rate
        beta = self.params[1]
        eps = 1e-3 # catch dividing by zero
        x_next = xp
        mean_square = np.zeros_like(xp)

        for i in range(num_iters):
            grad = self.surrogate.eval_gradient(x_next, surrogate_grad_params)
            mean_square = beta * mean_square + (1 - beta) * grad**2
            x_old = x_next
            x_next = x_next - alpha * grad / (np.sqrt(mean_square) + eps)

            if np.linalg.norm(x_next - x_old) < 1e-3:
                return x_next
        return x_next


class RMSPropBisection(Optimizer):
    """RMSProp with Bisection
    Do not use! It does not work well with Bisection method!
    """
    def __init__(self, surrogate, params=[1e-5, 0.9]):
        super().__init__(surrogate, params)

    def run(self, xp, num_iters, surrogate_grad_params, k=1):
        alpha = self.params[0]  # learning rate
        beta = self.params[1]
        eps = 1e-3  # catch dividing by zero
        x_next = xp
        mean_square = np.zeros_like(xp)
        grad = self.surrogate.eval_gradient(x_next, surrogate_grad_params)

        for i in range(1, num_iters):
            # store old point with its gradient
            x_old = x_next
            grad_old = grad

            # RMSProp update rule
            mean_square = beta * mean_square + (1 - beta) * grad_old ** 2
            x_new = x_next - alpha * grad_old / (np.sqrt(mean_square) + eps)
            grad = self.surrogate.eval_gradient(x_next, surrogate_grad_params)

            # Bisection
            # sign changed happened if sign is negative
            sign = grad * grad_old
            sign_changed = sign < 0
            # adapt alpha learning rate
            alpha = sign_changed * alpha / (k**2) + (1 - sign_changed) * alpha * k
            # x_next lies between x_old and x_new
            x_next = (1 - sign_changed) * x_new + sign_changed * (x_new + x_old) / 2
            #print(x_next)
            if np.linalg.norm(x_next - x_old) < 1e-3:
                return x_next
        return x_next
