# Python Standard Libraries
import numpy as np

# grAdapt package
from .base import Escape, inside_bounds, sample_points_bounds, bounds_range_ndim

# TODO
class CMAES(Escape):

    def __init__(self, surrogate, sampling_method=None):
        super().__init__(surrogate, sampling_method)

    def get_point(self, x_train, y_train, iteration, bounds):
        raise NotImplementedError('CMAES has not been implemented yet.')
