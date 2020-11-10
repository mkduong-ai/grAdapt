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
        bounds = [(-10, 10) for _ in range(5)]
        n1 = 100
        
        # first experiment
        model = Sequential(random_state=1234)
        model.minimize(function, bounds, n1, show_progressbar=False)
        res = model.res
        
        x_train1 = res['x']
        y_train1 = res['y']

        # second experiment
        model = Sequential(random_state=1234)
        model.minimize(function, bounds, n1, show_progressbar=False)
        res = model.res
        
        x_train2 = res['x']
        y_train2 = res['y']
        
        
        return x_train1, x_train2, y_train1, y_train2

    def test_equal_sphere(self):
        x_train1, x_train2, y_train1, y_train2 = self.equal_function(function=self.sphere)
        self.assertTrue(np.array_equal(x_train1, x_train2))
        self.assertTrue(np.array_equal(y_train1, y_train2))
        
    def test_equal_rastrigin(self):
        x_train1, x_train2, y_train1, y_train2 = self.equal_function(function=self.rastrigin)
        self.assertTrue(np.array_equal(x_train1, x_train2))
        self.assertTrue(np.array_equal(y_train1, y_train2))
        

if __name__ == '__main__':
    unittest.main()
