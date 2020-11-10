import unittest
from grAdapt.sampling.equidistributed import Density, KDistance, MaximalMinDistance, Mitchell, Reassignment
from grAdapt.utils.sampling import inside_bounds_ndim

import numpy as np


class TestEquidistributed(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self.sampling_methods = [Density(), KDistance(), MaximalMinDistance(), Mitchell(), Reassignment()]
        super().__init__(*args, **kwargs)

    @staticmethod
    def random_range(low, high, size):
        return np.random.rand(size)*(high-low)+low

    def test_inside_bounds(self):
        """
        Test all sampling methods in self.sampling_methods.
        Tests for several dimension and sample sizes
        whether each row is inside the given bound.
        """

        n_samples = np.random.randint(1, 128, 32)
        dims = np.random.randint(1, 20, 8)

        for sampling_method in self.sampling_methods:
            for i in n_samples:
                for d in dims:
                    lows = self.random_range(-10000, 0, d)
                    highs = self.random_range(0, 10000, d)
                    bounds = [(low, high) for low, high in zip(lows, highs)]
                    sample = sampling_method.sample(bounds, i)
                    self.assertEqual(True, inside_bounds_ndim(bounds, sample))
                    del sample


if __name__ == '__main__':
    unittest.main()
