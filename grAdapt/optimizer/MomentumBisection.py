# Python Standard Libraries
from abc import abstractmethod

# Third party imports
import numpy as np

from .base import Optimizer


class Momentum(Optimizer):
    """Optimizer object

    Implementation of Momentum (Polyak, 1964)

    """
    def __init__(self, surrogate, params=[1e-3, 0.9]):
        super().__init__(surrogate, params)

    def run(self, xp, num_iters, surrogate_grad_params):
        alpha = self.params[0]
        beta = self.params[1]
        x_next = xp
        mt = 0

        for i in range(num_iters):
            # grad has shape (d, )
            grad = self.surrogate.eval_gradient(x_next, surrogate_grad_params)
            x_old = x_next
            mt = beta * mt + (1 - beta) * grad
            x_next = x_next - alpha * mt

            # convergence
            if np.linalg.norm(x_old - x_next) < 1e-3:
                return x_next
        return x_next


class MomentumBisection(Optimizer):
    """Optimizer object

    Implementation of Momentum (Polyak, 1964)
    Modified with bisection method if a sign change of gradient happened

    """

    def __init__(self, surrogate, params=[1e-3, 0.9]):
        super().__init__(surrogate, params)

    def run(self, xp, num_iters, surrogate_grad_params, k=2):
        alpha = np.ones_like(xp) * self.params[0]
        beta = self.params[1]
        mt = 0
        x_next = xp
        grad = self.surrogate.eval_gradient(xp, surrogate_grad_params)

        for i in range(1, num_iters):
            # store old point with its gradient
            x_old = x_next
            grad_old = grad
            # Momentum update rule
            mt = beta * mt + (1 - beta) * grad_old
            x_new = x_next - alpha * mt
            grad = self.surrogate.eval_gradient(x_new, surrogate_grad_params)

            # Bisection
            # sign changed happened if sign is negative
            sign = grad * grad_old
            sign_changed = sign < 0
            # adapt alpha learning rate
            alpha = sign_changed * alpha / (k**2) + (1 - sign_changed) * alpha * k
            # x_next lies between x_old and x_new
            x_next = (1 - sign_changed) * x_new + sign_changed * (x_new + x_old) / 2
            # convergence
            if np.linalg.norm(x_old - x_next) < 1e-3:
                return x_next
        return x_next
