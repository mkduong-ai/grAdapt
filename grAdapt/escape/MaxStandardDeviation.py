# Python Standard Libraries
import numpy as np

# grAdapt package
from .base import Escape, inside_bounds, sample_points_bounds, bounds_range_ndim


class MaxStandardDeviation(Escape):

    def __init__(self, surrogate, sampling_method=None):
        super().__init__(surrogate, sampling_method)

    def get_point(self, x_train, y_train, iteration, bounds):
        # ignore warnings numerical issues
        import warnings
        warnings.filterwarnings('ignore')

        x_best = x_train[np.argmin(y_train)]  # x_best is current optimizer and "trust region"
        bounds_vec = ((bounds_range_ndim(bounds)/6)**2) / (np.log(iteration + np.e))  # shrink region to discover
        cov_matrix = np.diag(bounds_vec)

        # TODO: Points outside of bounds resample uniformly
        best_point = None
        best_std = 0

        # Many points created near current best x point
        # Pick the one where the standard deviation is highest
        for i in range(20):

            current_point = np.random.multivariate_normal(x_best, cov_matrix)  # (d, )
            _, current_std = self.surrogate.predict(current_point.reshape(1, -1), return_std=True)

            # max std and if is inside bounds
            if current_std >= best_std and inside_bounds(bounds, current_point):
                best_std = current_std
                best_point = current_point

        if best_point is None:
            return self.escape_history(bounds, x_train)
        else:
            return best_point

