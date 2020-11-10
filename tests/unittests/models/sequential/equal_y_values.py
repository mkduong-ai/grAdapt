import unittest
from grAdapt.models import Sequential

import numpy as np


class TestEqualYValues(unittest.TestCase):

    @staticmethod
    def rastrigin(x):
        x = np.array(x)
        return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=0)

    @staticmethod
    def sphere(x):
        return np.sum(x ** 2)

    def test_equal_sphere(self):
        model = Sequential()
        bounds = [(-10, 10) for _ in range(5)]
        n = 50
        res = model.minimize(self.sphere, bounds, n, show_progressbar=False)
        x_train = res['x']
        y_train = res['y']

        y_compare = np.array(list(map(self.sphere, x_train)))

        self.assertTrue(np.array_equal(y_compare, y_train))

    def test_equal_rastrigin(self):
        model = Sequential()
        bounds = [(-10, 10) for _ in range(5)]
        n = 50
        res = model.minimize(self.rastrigin, bounds, n, show_progressbar=False)
        x_train = res['x']
        y_train = res['y']

        y_compare = np.array(list(map(self.rastrigin, x_train)))

        self.assertTrue(np.array_equal(y_compare, y_train))


if __name__ == '__main__':
    unittest.main()
