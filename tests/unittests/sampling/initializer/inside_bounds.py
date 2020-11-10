import unittest
import grAdapt
from grAdapt.sampling.initializer import Standard, Vertices, VerticesForce, VerticesForceRandom
from grAdapt.sampling.equidistributed import MaximalMinDistance
from grAdapt.utils.sampling import inside_bounds_ndim


class TestInitalizer(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        sampling_method = MaximalMinDistance()
        self.all_initializers = [Standard(sampling_method),
                                 Vertices(sampling_method),
                                 VerticesForce(sampling_method),
                                 VerticesForceRandom(sampling_method)]
        super().__init__(*args, **kwargs)

    def test_inside_bounds(self):
        """
        Test all Initializers in all_initializers.
        Tests for several dimension and sample sizes
        whether each row is inside the given bound.
        """

        n_samples = 32
        dim = 32

        for initializer in self.all_initializers:
            for i in range(1, n_samples):
                for d in range(1, dim):
                    bounds = [(-10, 10) for i in range(d)]
                    sample = initializer.sample(bounds, i)
                    self.assertEqual(True, inside_bounds_ndim(bounds, sample))


if __name__ == '__main__':
    unittest.main()
