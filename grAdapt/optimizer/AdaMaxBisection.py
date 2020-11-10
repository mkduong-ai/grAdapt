# Python Standard Libraries
from abc import abstractmethod

# Third party imports
import numpy as np

from .base import Optimizer


class AdaMax(Optimizer):
    """Optimizer object

    Implementation of AdaMax (Kingma and Ba, 2015)

    """
    def __init__(self, surrogate, params=[0.002, 0.9, 0.999]):
        super().__init__(surrogate, params)

    def run(self, xp, num_iters, surrogate_grad_params):
        alpha = self.params[0]
        beta1 = self.params[1]
        beta2 = self.params[2]
        x_next = xp
        mt = 0
        vt = 0
        eps = 1e-6

        for i in range(num_iters):
            # grad has shape (d, )
            grad = self.surrogate.eval_gradient(x_next, surrogate_grad_params)
            x_old = x_next

            # update rule
            mt = beta1 * mt + (1 - beta1) * grad
            vt = np.maximum(beta2*vt, np.abs(grad))
            mt_hat = mt / (1 - beta1 ** i)
            x_next = x_next - (alpha / (vt + eps)) * mt_hat

            # convergence
            if np.linalg.norm(x_old - x_next) < 1e-3:
                return x_next
        return x_next


class AdaMaxBisection(Optimizer):
    """Optimizer object

    Implementation of AdaMax (Kingma and Ba, 2015)
    Modified with bisection method if a sign change of gradient happened

    """

    def __init__(self, surrogate, params=[0.002, 0.9, 0.999]):
        super().__init__(surrogate, params)

    def run(self, xp, num_iters, surrogate_grad_params, k=1):
        alpha = self.params[0]
        beta1 = self.params[1]
        beta2 = self.params[2]
        x_next = xp
        mt = 0
        vt = 0
        eps = 1e-6
        grad = self.surrogate.eval_gradient(xp, surrogate_grad_params)

        for i in range(1, num_iters):
            # store old point with its gradient
            x_old = x_next
            grad_old = grad

            # AdaMax update rule
            mt = beta1 * mt + (1 - beta1) * grad_old
            vt = np.maximum(beta2 * vt, np.abs(grad_old))
            mt_hat = mt / (1 - beta1 ** i)
            x_new = x_next - (alpha / (vt + eps)) * mt_hat

            # gradient update
            grad = self.surrogate.eval_gradient(xp, surrogate_grad_params)

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
