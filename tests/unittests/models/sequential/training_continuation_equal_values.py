import unittest
from grAdapt.models import Sequential

import numpy as np


class TestTrainingContinuationEqualValues(unittest.TestCase):

    @staticmethod
    def rastrigin(x):
        x = np.array(x)
        return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=0)

    @staticmethod
    def sphere(x):
        return np.sum(x ** 2)

    @staticmethod
    def equal_function(function):
        model = Sequential()
        bounds = [(-10, 10) for _ in range(5)]
        n1 = 30
        n2 = 50
        model.minimize(function, bounds, n1, show_progressbar=False)
        res = model.minimize(function, bounds, n2, show_progressbar=False)

        x_train = res['x']
        y_train = res['y']

        y_compare = np.array(list(map(function, x_train)))
        
        print('x_train:')
        print(x_train)
        print('y_sol')
        print(res['y_sol'])

        return y_train, y_compare, x_train

    def test_equal_sphere(self):
        y_train, y_compare, x_train = self.equal_function(function=self.sphere)
        self.assertTrue(np.array_equal(y_compare, y_train))

    def test_equal_rastrigin(self):
        y_train, y_compare, x_train = self.equal_function(function=self.rastrigin)
        self.assertTrue(np.array_equal(y_compare, y_train))


if __name__ == '__main__':
    unittest.main()
