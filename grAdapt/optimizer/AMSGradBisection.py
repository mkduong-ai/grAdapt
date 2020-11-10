# Python Standard Libraries
from abc import abstractmethod

# Third party imports
import numpy as np

from .base import Optimizer


class AMSGrad(Optimizer):
    """Optimizer object

    Implementation of AMSGrad (Reddi et al., 2018)

    """
    def __init__(self, surrogate, params=[0.001, 0.9, 0.999, 1e-7]):
        super().__init__(surrogate, params)

    def run(self, xp, num_iters, surrogate_grad_params):
        alpha = self.params[0]
        beta1 = self.params[1]
        beta2 = self.params[2]
        eps = self.params[3]
        x_next = xp
        mt = np.zeros_like(xp)
        vt = np.zeros_like(xp)
        vt_hat = np.zeros_like(xp)

        for i in range(num_iters):
            # grad has shape (d, )
            grad = self.surrogate.eval_gradient(x_next, surrogate_grad_params)
            x_old = x_next

            # update rule
            mt = beta1 * mt + (1 - beta1) * grad
            vt = beta2 * vt + (1 - beta2) * (grad ** 2)
            vt_hat = np.maximum(vt_hat, vt)
            x_next = x_next - (alpha / (np.sqrt(vt_hat) + eps)) * mt

            # convergence
            if np.linalg.norm(x_old - x_next) < 1e-3:
                return x_next
        return x_next


class AMSGradBisection(Optimizer):
    """Optimizer object

    Implementation of AMSGrad (Reddi et al., 2018)
    Modified with bisection method if a sign change of gradient happened

    """

    def __init__(self, surrogate, params=[0.001, 0.9, 0.999, 1e-7]):
        super().__init__(surrogate, params)

    def run(self, xp, num_iters, surrogate_grad_params, k=1):
        alpha = self.params[0]
        beta1 = self.params[1]
        beta2 = self.params[2]
        eps = self.params[3]
        x_next = xp
        mt = np.zeros_like(xp)
        vt = np.zeros_like(xp)
        vt_hat = np.zeros_like(xp)
        grad = self.surrogate.eval_gradient(xp, surrogate_grad_params)

        for i in range(1, num_iters):
            # store old point with its gradient
            x_old = x_next
            grad_old = grad

            # AMSGrad update rule
            mt = beta1 * mt + (1 - beta1) * grad_old
            vt = beta2 * vt + (1 - beta2) * (grad_old ** 2)
            vt_hat = np.maximum(vt_hat, vt)
            x_new = x_next - (alpha / (np.sqrt(vt_hat) + eps)) * mt

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
