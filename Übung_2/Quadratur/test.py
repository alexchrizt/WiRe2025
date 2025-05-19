import unittest
import numpy as np
from Übung_2.Quadratur.quadrature import newton_cotes_weights, newton_cotes_quadrature, global_newton_cotes_quadrature, monte_carlo_quadrature

class TestInterpolation(unittest.TestCase):
    # These tests check whether or not input and output has the correct
    # format for all methods that are to be implemented.
    # These are not the 'real' tests for the assessment!
    def test_Newton_Cotes_Weights(self):
        # The real test will be similar, i.e., testing even more
        # quadrature rules. So feel free to extend this test to
        # different quadrature rules.
        weights = newton_cotes_weights(1)
        self.assertTrue(np.allclose(weights, [.5, .5]))
    
    def test_local_Newton_Cotes(self):
        f = lambda x: np.cos(x) + np.exp(x) + 3 * x ** 2
        Ix = newton_cotes_quadrature(f, 0, 5, 7)
        self.assertEqual(Ix, Ix)

    def test_global_Newton_Cotes(self):
        f = lambda x: np.cos(x) + np.exp(x) + 3 * x ** 2
        Ix = global_newton_cotes_quadrature(f, 0, 5, 7, 10)
        self.assertEqual(Ix, Ix)

    def test_Monte_Carlo(self):
        f = lambda x: np.cos(x) + np.exp(x) + 3 * x ** 2
        np.random.seed(42)  # Fixed seed for reproducibility
        Ix = monte_carlo_quadrature(f, 0, 5, 1000)
        self.assertEqual(Ix, Ix)


if __name__ == '__main__':
    unittest.main()
