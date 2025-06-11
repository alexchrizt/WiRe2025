import unittest
import warnings
import numpy as np
from lgs import check_input, gauss, frobenius_norm

class TestGauss(unittest.TestCase):
    def test_check_input(self):
        with self.subTest("Test for error when length of b does not match shape of A"):
            A = np.array([[2, 1], [3, 4]])
            b = np.array([1, 2, 3])
            self.assertRaises(ValueError, check_input, A, b)
        with self.subTest("Check for Warning if A is not diagonal dominant"):
            A = np.array([[1, 2], [3, 4]])
            b = np.array([1, 5])
            with self.assertWarns(Warning):
                check_input(A, b)




if __name__ == '__main__':
    unittest.main()
