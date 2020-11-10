# Python Standard Libraries
from abc import abstractmethod

# Third party imports
import numpy as np

from .base import Optimizer


class Adam(Optimizer):
    """Optimizer object

    Implementation of ADAM (Kingma et al.)

    """

    def __init__(self, surrogate, params=[1e-3, 0.9, 0.999, 1e-8]):
        super().__init__(surrogate, params)

    def run(self, xp, num_iters, surrogate_grad_params):
        alpha = self.params[0]
        beta_1 = self.params[1]
        beta_2 = self.params[2]
        epsilon = self.params[3]
        m_t = 0
        v_t = 0
        x_next = xp

        for t in range(1, num_iters + 1):
            g_t = self.surrogate.eval_gradient(x_next, surrogate_grad_params)
            x_old = x_next

            # adam update rule
            m_t = beta_1 * m_t + (1 - beta_1) * g_t
            v_t = beta_2 * v_t + (1 - beta_2) * (g_t * g_t)
            m_cap = m_t / (1 - (beta_1 ** t))
            v_cap = v_t / (1 - (beta_2 ** t))
            x_next = x_next - (alpha * m_cap) / (np.sqrt(v_cap) + epsilon)  # update step

            if np.linalg.norm(x_next - x_old) < 1e-3:
                return x_next
        return x_next


class AdamBisection(Optimizer):
    """Optimizer object

    Implementation of ADAM (Kingma et al.)
    Modified with bisection method if a sign change of gradient happened

    """

    def __init__(self, surrogate, params=[1e-3, 0.9, 0.999, 1e-8]):
        super().__init__(surrogate, params)

    def run(self, xp, num_iters, surrogate_grad_params, k=1):
        alpha = np.ones_like(xp) * self.params[0]
        beta_1 = self.params[1]
        beta_2 = self.params[2]
        epsilon = self.params[3]
        m_t = 0
        v_t = 0
        x_next = xp
        g_t = self.surrogate.eval_gradient(x_next, surrogate_grad_params)

        for t in range(2, num_iters + 1):
            # store old point with its gradient
            x_old = x_next
            g_t_old = g_t

            # ADAM update rule
            m_t = beta_1 * m_t + (1 - beta_1) * g_t_old
            v_t = beta_2 * v_t + (1 - beta_2) * (g_t_old * g_t_old)
            m_cap = m_t / (1 - (beta_1 ** t))
            v_cap = v_t / (1 - (beta_2 ** t))
            x_new = x_next - (alpha * m_cap) / (np.sqrt(v_cap) + epsilon)  # update step

            # gradient update
            g_t = self.surrogate.eval_gradient(x_new, surrogate_grad_params)

            # Bisection
            # sign changed happened if sign is negative
            sign = g_t * g_t_old
            sign_changed = sign < 0
            # adapt alpha learning rate
            alpha = sign_changed * alpha / (k**2) + (1 - sign_changed) * alpha * k
            # x_next lies between x_old and x_new
            x_next = (1 - sign_changed) * x_new + sign_changed * (x_new + x_old) / 2
            # convergence
            if np.linalg.norm(x_next - x_old) < 1e-3:
                return x_next
        return x_next
