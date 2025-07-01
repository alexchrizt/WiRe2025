import unittest
import numpy as np
from newton import newton, newton_global

class TestNewton(unittest.TestCase):
    def test_newton(self):
        with self.subTest("Check for correct length of output vector"):
            f = lambda x: x**2
            df = lambda x: 2*x
            x_k = newton(f, df, 0.0)
            self.assertEqual(len(x_k), 1)



if __name__ == '__main__':
    unittest.main()
