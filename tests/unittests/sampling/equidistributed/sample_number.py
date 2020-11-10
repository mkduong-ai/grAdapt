import unittest
from grAdapt.sampling.equidistributed import Density, KDistance, MaximalMinDistance, Mitchell, Reassignment
from grAdapt.utils.sampling import inside_bounds_ndim

import numpy as np


class TestEquidistributed(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self.sampling_methods = [Density(), KDistance(), MaximalMinDistance(), Mitchell(), Reassignment()]
        super().__init__(*args, **kwargs)

    def test_sample_number(self):
        """
        Test all sampling methods in self.sampling_methods.
        Tests for several dimension and sample sizes
        whether the first-axis of the output equals
        the sample size n.
        """
        n_samples = 33
        dims = np.random.randint(1, 40, 10)

        for sampling in self.sampling_methods:
            for i in range(1, n_samples):
                for d in dims:
                    bounds = [(-10, 10) for i in range(d)]
                    sample = sampling.sample(bounds, i)
                    self.assertEqual(sample.shape[0], i)


if __name__ == '__main__':
    unittest.main()
