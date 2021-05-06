# Python Standard Libraries
import warnings
import time
import os
import sys
from pathlib import Path

# Third party imports
# fancy prints
import numpy as np
from tqdm import tqdm

# grAdapt package
import grAdapt.utils.math
import grAdapt.utils.misc
import grAdapt.utils.sampling
from grAdapt import surrogate as sur, optimizer as opt, escape as esc
from grAdapt.space.transformer import Transformer
from grAdapt.sampling import initializer as init, equidistributed as equi


class Asynchronous:
    def __init__(self, bounds, surrogate=None, optimizer=None, sampling_method=None,
                 escape=None, training=None, random_state=1,
                 n_evals='auto', eps=1e-3, f_min=-np.inf, f_min_eps=1e-2, n_random_starts='auto',
                 auto_checkpoint=False, show_progressbar=True, prints=True):
        """
        Parameters
        ----------
        bounds : list
             list of tuples e.g. [(-5, 5), (-5, 5)]
        surrogate : grAdapt Surrogate object
        optimizer : grAdapt Optimizer object
        sampling_method : Sampling Method to be used. static method from utils
        escape : grAdapt Escape object
        training : (X, y) with X shape (n, m) and y shape (n,)
        random_state : integer
            random_state integer sets numpy seed
        bounds : list
             list of tuples e.g. [(-5, 5), (-5, 5)]
        """
        # Stock module settings
        self.bounds = bounds

        # seed
        self.random_state = random_state
        np.random.seed(self.random_state)

        if surrogate is None:
            self.surrogate = sur.GPRSlidingWindow()
        else:
            self.surrogate = surrogate

        if optimizer is None:
            self.optimizer = opt.AMSGradBisection(surrogate=self.surrogate)
        else:
            self.optimizer = optimizer
            if surrogate is None:
                raise Exception('If optimizer is passed, then surrogate must be passed, too.')

        if sampling_method is None:
            self.sampling_method = equi.MaximalMinDistance()
        else:
            self.sampling_method = sampling_method

        if escape is None:
            self.escape = esc.NormalDistributionDecay(surrogate=self.surrogate, sampling_method=self.sampling_method)
        else:
            self.escape = escape
            if surrogate is None or sampling_method is None:
                raise Exception('When passing an escape function, surrogate and sampling_method must be passed, too.')

        # other settings
        # continue optimizing
        self.training = training
        if training is not None:
            self.X = list(training[0])
            self.y = list(training[1])
            if len(self.X) != len(self.y):
                raise AssertionError('Training data not valid. Length of X and y must be the same.')
            # self.fit(self.X, self.y)
        else:
            self.X = list(grAdapt.utils.sampling.sample_points_bounds(self.bounds, 11))
            self.y = []

        self.n_evals = n_evals
        self.eps = eps
        self.f_min = f_min
        self.f_min_eps = f_min_eps
        self.n_random_starts = n_random_starts

        # keep track of checkpoint files
        self.checkpoint_file = None
        self.auto_checkpoint = auto_checkpoint

        # results
        self.res = None

        self.show_progressbar = show_progressbar
        self.prints = prints

        # save current iteration
        if training is not None:
            self.iteration = len(self.X) - 1
        else:
            self.iteration = 0

    def escape_x_criteria(self, x_train, iteration):
        """Checks whether new point is different than the latest point by the euclidean distance
        Checks whether new point is inside the defined search space/bounds.
        Returns True if one of the conditions above are fulfilled.

        Parameters
        ----------
        x_train : ndarray (n, d)
        iteration : integer

        Returns
        -------
        boolean
        """
        # x convergence
        # escape_convergence = (np.linalg.norm(x_train[iteration - 1] - x_train[iteration])) < self.eps
        n_hist = 2
        escape_convergence_history = any(
            (np.linalg.norm(x_train[iteration - (n_hist + 1):] - x_train[iteration - 1], axis=1)) < self.eps)

        # check whether point is inside bounds
        escape_valid = not (grAdapt.utils.sampling.inside_bounds(self.bounds, x_train[iteration - 1]))

        # escape_x = escape_convergence or escape_valid
        escape_x = escape_convergence_history or escape_valid
        return escape_x

    @staticmethod
    def escape_y_criteria(y_train, iteration, pct):
        """

        Parameters
        ----------
        y_train : array-like (n, d)
        iteration : integer
        pct : numeric
            pct should be less than 1.

        Returns
        -------
        boolean
        """
        try:
            return grAdapt.utils.misc.is_inside_relative_range(y_train[iteration - 1], y_train[iteration - 2], pct)
        except:
            return False

    def dummy(self):
        return 0

    def ask(self):
        if len(self.X) > len(self.y):  # initial points
            self.iteration += 1

            # if user asks consecutively without telling
            if self.iteration == len(self.y) + 2:
                self.iteration -= 1
                warnings.warn("Tell the optimizer/model after you ask.", RuntimeWarning)

            return self.X[self.iteration - 1]

        else:
            # gradient parameters specific for the surrogate model
            surrogate_grad_params = [np.array(self.X[:self.iteration]), np.array(self.y[:self.iteration]),
                                     self.dummy, self.bounds]

            # apply optimizer
            return_x = self.optimizer.run(self.X[self.iteration - 1],
                                          grAdapt.utils.misc.epochs(self.iteration),
                                          surrogate_grad_params)

            # escape indicator variables
            escape_x_criteria_boolean = self.escape_x_criteria(np.array(self.X), self.iteration)
            escape_y_criteria_boolean = self.escape_y_criteria(self.y, self.iteration, self.f_min_eps)
            escape_boolean = escape_x_criteria_boolean or escape_y_criteria_boolean

            # sample new point if must escape or bounds not valid
            if escape_boolean:
                return_x = self.escape.get_point(self.X[:self.iteration], self.y[:self.iteration],
                                                 self.iteration, self.bounds)

            self.iteration += 1

            # save current training data

            return return_x

    def tell(self, next_x, f_val):
        if len(self.X) > len(self.y):
            # no need to append x
            self.y.append(f_val)
        elif len(self.X) == len(self.y):
            # append
            self.X.append(next_x)
            self.y.append(f_val)
        else:
            raise RuntimeError('More function values available than x values/parameter sets.')

        # Fit data on surrogate model
        self.surrogate.fit(np.array(self.X[:self.iteration]), np.array(self.X[:self.iteration]))
