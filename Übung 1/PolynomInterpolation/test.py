import unittest
import numpy as np
from interpolation import newton_equidistant, newton_chebyshev, cubic_spline_equidistant, cubic_spline_chebyshev

class TestInterpolation(unittest.TestCase):
    # These tests check whether or not input and output has the correct
    # format for all methods that are to be implemented.
    # These are not the 'real' tests for the assessment!
    def test_Newton_Equidistant(self):
        f = lambda x: np.exp(x)
        [x, y], [xx, yy] = newton_equidistant(f, [-2, 2], 10, 7)
        with self.subTest("Test Sample Points"):
            self.assertTrue(np.allclose(xx, xx))
        with self.subTest("Test Values at Sample Points"):
            self.assertTrue(np.allclose(yy, yy))
        with self.subTest("Test Evaluation Points"):
            self.assertTrue(np.allclose(x, x))
        with self.subTest("Test Values at Evaluation Points"):
            self.assertTrue(np.allclose(y, y))

    def test_Newton_Chebyshev(self):
        f = lambda x: np.exp(x)
        [x, y], [xx, yy] = newton_chebyshev(f, [-2, 2], 10, 7)
        with self.subTest("Test Sample Points"):
            self.assertTrue(np.allclose(xx, xx))
        with self.subTest("Test Values at Sample Points"):
            self.assertTrue(np.allclose(yy, yy))
        with self.subTest("Test Evaluation Points"):
            self.assertTrue(np.allclose(x, x))
        with self.subTest("Test Values at Evaluation Points"):
            self.assertTrue(np.allclose(y, y))

    def test_Cubic_Spline_Equidistant(self):
        f = lambda x: np.exp(x)
        [x, y], [xx, yy] = cubic_spline_equidistant(f, [-2, 2], 10, 7)
        with self.subTest("Test Sample Points"):
            self.assertTrue(np.allclose(xx, xx))
        with self.subTest("Test Values at Sample Points"):
            self.assertTrue(np.allclose(yy, yy))
        with self.subTest("Test Evaluation Points"):
            self.assertTrue(np.allclose(x, x))
        with self.subTest("Test Values at Evaluation Points"):
            self.assertTrue(np.allclose(y, y))

    def test_Cubic_Spline_Chebyshev(self):
        f = lambda x: np.exp(x)
        [x, y], [xx, yy] = cubic_spline_chebyshev(f, [-2, 2], 10, 7)
        with self.subTest("Test Sample Points"):
            self.assertTrue(np.allclose(xx, xx))
        with self.subTest("Test Values at Sample Points"):
            self.assertTrue(np.allclose(yy, yy))
        with self.subTest("Test Evaluation Points"):
            self.assertTrue(np.allclose(x, x))
        with self.subTest("Test Values at Evaluation Points"):
            self.assertTrue(np.allclose(y, y))



if __name__ == '__main__':
    unittest.main()
