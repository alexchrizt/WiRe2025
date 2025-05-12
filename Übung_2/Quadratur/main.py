# Automated check that you use the correct Python version
from sys import version_info

if version_info[0] < 3 or version_info[1] < 10:
    raise Exception("Must be using Python 3.10 or newer")
###########################################################
from scipy.interpolate import CubicSpline
from quadrature import newton_cotes_quadrature, global_newton_cotes_quadrature, monte_carlo_quadrature
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Implement the plotting and anything you need for it here
    print("Hello World!") 
