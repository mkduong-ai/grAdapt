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
from numba.typed import List

# grAdapt package
import grAdapt.utils.math
import grAdapt.utils.misc
import grAdapt.utils.sampling
from grAdapt import surrogate as sur, optimizer as opt, escape as esc
from grAdapt.space.transformer import Transformer
from grAdapt.sampling import initializer as init, equidistributed as equi


class Sequential:
    def __init__(self, surrogate=None, optimizer=None, sampling_method=None,
                 initializer=None, escape=None,
                 training=None, random_state=1):
        """
        Parameters
        ----------
        surrogate : grAdapt Surrogate object
        optimizer : grAdapt Optimizer object
        sampling_method : Sampling Method to be used. static method from utils
        initializer : grAdapt Initializer object
        escape : grAdapt Escape object
        training : (X, y) with X shape (n, m) and y shape (n,)
        random_state : integer
            random_state integer sets numpy seed
        """

        # Standard Values
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

        if initializer is None:
            self.initializer = init.VerticesForceRandom(sampling_method=self.sampling_method)
        else:
            self.initializer = initializer
            if sampling_method is None:
                raise Exception('If initializer is passed, then sampling_method must be passed, too.')

        if escape is None:
            self.escape = esc.NormalDistributionDecay(surrogate=self.surrogate, sampling_method=self.sampling_method)
        else:
            self.escape = escape
            if surrogate is None or sampling_method is None:
                raise Exception('When passing an escape function, surrogate and sampling_method must be passed, too.')

        # continue optimizing
        self.training = training
        if training is not None:
            self.X = training[0]
            self.y = training[1]
            # self.fit(self.X, self.y)
        else:
            self.X = None
            self.y = None

        # seed
        self.random_state = random_state
        np.random.seed(self.random_state)

        # results
        self.res = None

        # keep track of checkpoint files
        self.checkpoint_file = None
        self.auto_checkpoint = False

    def warn_n_evals(self):
        if self.n_evals <= 0:
            raise Exception('Please set n_evals higher higher than 0.')

        if self.n_evals <= self.dim:
            warnings.warn('n_evals should be higher than the dimension of the problem.')

        if self.n_random_starts == 'auto' or not isinstance(self.n_random_starts, (int, float)):
            self.n_random_starts = grAdapt.utils.misc.random_starts(self.n_evals, self.dim)

        if self.n_evals < self.n_random_starts:
            warnings.warn('n_random starts can\'t be higher than n_evals.')
            warnings.warn('n_random_starts set automatically. ')
            self.n_random_starts = grAdapt.utils.misc.random_starts(self.n_evals, self.dim)

    def fit(self, X, y):
        """Fit known points on surrogate model

        Parameters
        ----------
        X : array-like (n, m)
        y : array (n,)

        Returns
        -------
        None

        """
        print('Surrogate model fitting on known points.')
        self.surrogate.fit(X, y)

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
        escape_convergence = (np.linalg.norm(x_train[iteration - 1] - x_train[iteration])) < self.eps
        # check whether point is inside bounds
        escape_valid = not (grAdapt.utils.sampling.inside_bounds(self.bounds, x_train[iteration]))

        escape_x = escape_convergence or escape_valid
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

    def minimize(self, func, bounds, n_evals='auto', eps=1e-3, f_min=-np.inf, f_min_eps=1e-2, n_random_starts='auto',
                 auto_checkpoint=False, show_progressbar=True, prints=True):
        """

        Parameters
        ----------
        func : takes ndarray and returns scalar
        bounds : list
             list of tuples e.g. [(-5, 5), (-5, 5)]
        n_evals : int or string
            number of max. function evaluations
            'auto' : will evaluate func based on the probability of
                    hitting the optimal solution by coincidence
                    between 100 and 10000
        eps : float
            convergence criteria, absolute tolerance
        f_min : numeric
            if the minimal target value of func is known
            can lead to earlier convergence
        f_min_eps : numeric
            early stop criteria, relative tolerance
        n_random_starts : string or int
        auto_checkpoint : bool
        show_progressbar : bool
        prints : bool

        Returns
        -------
        res: dictionary
            res['x'] : ndarray (n, d)
            res['y'] : ndarray (n,)
            res['x_sol'] : solution vector
            res['y_sol'] : y solution
            res['surrogate'] : grAdapt surrogate object

        """
        # print('Optimizing ' + func.__name__)
        self.func = Transformer(func, bounds)
        #self.bounds = List()  # Numba Typed List for performance
        #[self.bounds.append((float(x[0]), float(x[1]))) for x in bounds]
        self.bounds = bounds
        self.n_evals = int(n_evals)
        self.eps = eps
        self.f_min = f_min
        self.f_min_eps = f_min_eps
        self.n_random_starts = n_random_starts
        self.auto_checkpoint = auto_checkpoint
        self.dim = len(bounds)
        self.prints = prints

        if self.prints is False:
            sys.stdout = open(os.devnull, 'w')
        """n_evals value based on the probability of finding
        the optimal solution by coincidence
        """
        if self.n_evals == 'auto':
            vol_sphere = grAdapt.utils.math.geometry.volume_hypersphere(len(bounds), self.eps)
            vol_rec = grAdapt.utils.math.geometry.volume_hyperrectangle(bounds)

            # limit n_evals to 100 and 10000
            self.n_evals = max(100, int(vol_rec / vol_sphere))
            self.n_evals = min(10000, self.n_evals)

        """Catching errors/displaying warnings related to n_evals and n_random_starts
        and automatically set n_random_starts if not given
        """
        self.warn_n_evals()

        """checkpoint print directory
        """
        if auto_checkpoint:
            directory_path = os.getcwd() + '/checkpoints'
            print('auto_checkpoint set to True. The training directory is located at\n' + directory_path)

        """Inittialize x_train and y_train
        Check whether training can be continued
        """
        # TODO: Change number of dimensions for nominal datatype
        x_train = np.zeros((self.n_evals, self.dim))
        y_train = np.zeros((self.n_evals,)) + np.inf
        if self.training is not None:
            x_train = np.vstack((self.X, x_train))
            y_train = np.hstack((self.y, y_train))

            # prevent same sample points in training continuation
            self.random_state += 1
            np.random.seed(self.random_state)
            print('Training data added successfully.')

        """Randomly guess n_random_starts points
        """
        print('Sampling {0} random points.'.format(self.n_random_starts))
        print('Random function evaluations. This might take a while.')
        train_len = len(y_train) - self.n_evals

        if train_len == 0:
            x_train[train_len:self.n_random_starts + train_len] = \
                self.initializer.sample(self.bounds, self.n_random_starts)
            y_train[train_len:self.n_random_starts + train_len] = \
                np.array(list(map(self.func, tqdm(x_train[train_len:self.n_random_starts + train_len],
                                                  total=self.n_evals, leave=False))))
        else:
            x_train[train_len:self.n_random_starts + train_len] = \
                self.sampling_method.sample(self.bounds, self.n_random_starts, x_history=x_train[:train_len])
            y_train[train_len:self.n_random_starts + train_len] = \
                np.array(list(map(self.func, tqdm(x_train[train_len:self.n_random_starts + train_len],
                                                  total=self.n_evals, leave=False))))

        print('Finding optimum...')

        """Start from best point
        """

        best_idx = np.argmin(y_train[:self.n_random_starts + train_len])

        # swap positions
        x_train[[best_idx, self.n_random_starts - 1 + train_len]] = \
            x_train[[self.n_random_starts - 1 + train_len, best_idx]]
        y_train[[best_idx, self.n_random_starts - 1 + train_len]] = \
            y_train[[self.n_random_starts - 1 + train_len, best_idx]]

        """Optimizing loop
        """
        start_time = time.perf_counter()
        pbar = tqdm(total=self.n_evals + train_len, initial=self.n_random_starts + train_len,
                    disable=not show_progressbar)
        for iteration in range(self.n_random_starts + train_len, self.n_evals + train_len):
            # print(iteration)
            pbar.update()  # progressbar update

            # Fit data on surrogate model
            self.surrogate.fit(x_train[:iteration], y_train[:iteration])

            # gradient parameters specific for the surrogate model
            surrogate_grad_params = [x_train[:iteration], y_train[:iteration], self.func, bounds]
            # print(x_train[iteration-1])
            x_train[iteration] = self.optimizer.run(x_train[iteration - 1], grAdapt.utils.misc.epochs(iteration),
                                                    surrogate_grad_params)

            escape_x_criteria_boolean = self.escape_x_criteria(x_train, iteration)
            escape_y_criteria_boolean = self.escape_y_criteria(y_train, iteration, self.f_min_eps)
            escape_boolean = escape_x_criteria_boolean or escape_y_criteria_boolean

            # sample new point if must escape or bounds not valid
            if escape_boolean:
                x_train[iteration] = self.escape.get_point(x_train[:iteration], y_train[:iteration], iteration,
                                                           self.bounds)

            # obtain y_train
            y_train[iteration] = self.func(x_train[iteration])

            # stop early

            if grAdapt.utils.misc.is_inside_relative_range(y_train[iteration], self.f_min, self.f_min_eps):
                break

            # auto_checkpoint
            if auto_checkpoint and time.perf_counter() - start_time >= 60:
                self.X = x_train
                self.y = y_train
                res = self.build_res()
                self.save_checkpoint(res)
                start_time = time.perf_counter()

        # progressbar
        pbar.close()

        self.X = x_train
        self.y = y_train
        # save current training data
        self.training = (self.X, self.y)
        self.res = self.build_res()

        if auto_checkpoint:
            self.save_checkpoint(self.res)

        # restore prints
        if self.prints is False:
            sys.stdout = sys.__stdout__

        return self.res

    def maximize(self, func, bounds, *args, **kwargs):
        """

        Parameters
        ----------
        func : takes ndarray and returns scalar
        bounds : list of tuples e.g. [(-5, 5), (-5, 5)]
        args : args from minimize
        kwargs : args from minimize

        Returns
        -------
        res: dictionary
            res['x'] : ndarray (n, d)
            res['y'] : ndarray (n,)
            res['x_sol'] : solution vector
            res['y_sol'] : y solution
            res['surrogate'] : grAdapt surrogate object
        """

        def f_max(x):
            return -func(x)

        # x_train, y_train, surrogate = self.minimize(f_max, bounds, *args, **kwargs)
        res = self.minimize(f_max, bounds, *args, **kwargs)
        res['y'] = -res['y']
        res['y_sol'] = -res['y_sol']

        # save the right y values
        if self.auto_checkpoint:
            self.save_checkpoint(res)

        return res

    def minimize_args(self, func, bounds, *args, **kwargs):
        """

        Parameters
        ----------
        func : takes ndarray and returns scalar
        bounds : list of tuples e.g. [(-5, 5), (-5, 5)]
        args : args from minimize
        kwargs : args from minimize

        Returns
        -------
        res: dictionary
            res['x'] : ndarray (n, d)
            res['y'] : ndarray (n,)
            res['x_sol'] : solution vector
            res['y_sol'] : y solution
            res['surrogate'] : grAdapt surrogate object
        """

        def f_min_args(*args2):
            arr = args2[0]
            args_list = arr.tolist()

            return func(*args_list)

        # x_train, y_train, surrogate = self.minimize(f_max, bounds, *args, **kwargs)
        res = self.minimize(f_min_args, bounds, *args, **kwargs)

        return res

    def maximize_args(self, func, bounds, *args, **kwargs):
        """

        Parameters
        ----------
        func : takes ndarray and returns scalar
        bounds : list of tuples e.g. [(-5, 5), (-5, 5)]
        args : args from minimize
        kwargs : args from minimize

        Returns
        -------
        res: dictionary
            res['x'] : ndarray (n, d)
            res['y'] : ndarray (n,)
            res['x_sol'] : solution vector
            res['y_sol'] : y solution
            res['surrogate'] : grAdapt surrogate object
        """

        def f_max(*args2):
            return -func(*args2)

        # x_train, y_train, surrogate = self.minimize(f_max, bounds, *args, **kwargs)
        res = self.minimize_args(f_max, bounds, *args, **kwargs)
        res['y'] = -res['y']
        res['y_sol'] = -res['y_sol']

        # save the right y values
        if self.auto_checkpoint:
            self.save_checkpoint(res)

        return res

    def scipy_wrapper_minimize(self, func, x0, bounds=None, tol=None, *args, **kwargs):
        """
        @ https://github.com/scipy/scipy/blob/v1.5.3/scipy/optimize/_minimize.py
        Parameters
        ----------
        func : callable
            The objective function to be minimized.
            ``fun(x, *args) -> float``
            where ``x`` is an 1-D array with shape (d,) and ``args``
            is a tuple of the fixed parameters needed to completely
            specify the function.
        x0 : ndarray, shape (d,)
            Initial guess. Array of real elements of size (d,),
        bounds : sequence or `Bounds`, optional
            There are two ways to specify the bounds:

                1. Instance of `Bounds` class.
                2. Sequence of ``(min, max)`` pairs for each element in `x`. None
                   is used to specify no bound.
        tol : float, optional
            Tolerance for termination. For detailed control, use solver-specific options.
        -------

        Returns
        -------
        ndarray (d,)
        """
        self.training = (x0.reshape(1, -1), func(x0))
        self.X = self.training[0]
        self.y = self.training[1]
        # self.fit(self.X, self.y)
        res = self.minimize(func, bounds, *args, **kwargs)

        return res['x_sol']

    def build_res(self):
        # define output
        transformed_input = []
        for i in range(self.X.shape[0]):
            arguments = np.zeros((self.X.shape[1],), dtype='O')
            for j in range(len(arguments)):
                try:
                    arguments[j] = self.bounds[j].transform(self.X[i, j])
                except:
                    arguments[j] = self.X[i, j]

            transformed_input.append(arguments)

        res = {'x': np.array(transformed_input), 'x_internal': self.X, 'y': self.y,
               'x_sol': np.array(transformed_input)[np.argmin(self.y)],
               'x_sol_internal': self.X[np.argmin(self.y)], 'y_sol': np.min(self.y),
               'surrogate': self.surrogate,
               'optimizer': self.optimizer}

        return res

    def load_checkpoint(self, filename):
        """

        Parameters
        ----------
        filename : string to filepath of checkpoint

        Returns
        -------
        None

        """
        res = np.load(filename, allow_pickle=True).item()

        self.X = res['x']
        self.y = res['y']

        return res
        # self.surrogate = res['surrogate']

    def save_checkpoint(self, res):
        """

        Parameters
        ----------
        res: dictionary
            res['x'] : ndarray (n, d)
            res['y'] : ndarray (n,)
            res['x_sol'] : solution vector
            res['y_sol'] : y solution
            res['surrogate'] : grAdapt surrogate object

        Returns
        -------
        None

        """
        directory = Path(os.getcwd()) / 'checkpoints'
        filename = ('checkpointXY' + time.strftime('%y%b%d-%H%M%S') + '.npy')
        filename = directory / filename

        if not os.path.exists(directory):
            os.makedirs(directory)

        # save new file
        np.save(filename, res)
        print('Checkpoint created in ' + str(filename))

        # delete last file
        if self.checkpoint_file is not None:
            os.remove(self.checkpoint_file)
            print('Old checkpoint file {0} deleted.'.format(self.checkpoint_file))
        self.checkpoint_file = filename
