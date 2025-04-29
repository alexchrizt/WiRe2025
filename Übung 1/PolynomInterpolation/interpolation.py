# Automated check that you use the correct Python version #
from sys import version_info
if version_info[0] < 3 or version_info[1] < 10:
    raise Exception("Must be using Python 3.10 or newer")
###########################################################

import numpy as np
from scipy.interpolate import CubicSpline
from typing import Callable

# CAUTION: The input arguments of the following method MUST NOT be changed.
# Any changes may cause the automated tests to fail and, hence, reduce
# the total score in the evaluation of your submission.
def newton_equidistant(func: Callable[[float], float], bnds: list[float], n_eval_pts: int, n_sample_pts: int):
    """The method shall calculate the Newton interpolation of the function 'func' in the interval 'bnds'
        using 'n_sample_pts' equidistant sample points and 'n_eval_pts' evaluation points.
        
    Parameters
    ----------
    func : Callable
        Function that shall be approximated
    bnds : list[float]
        Interval in which 'func' shall be interpolated
    n_eval_pts : int
        Amount of evalutation points
    n_sample_pts : int
        Amount of sample points
        
    Returns
    -------
    [x, y] : list[np.ndarray]
        Evaluation points and values
    [xx, yy] : list[np.ndarray]
        Sample points and values
    """
    a, b = bnds
    xx = np.linspace(a, b, n)
    yy = func(xx)
    co = yy.astype(float).copy()

    for j in range(1, n_sample_pts):
        for i in range(n_sample_pts - 1, j - 1, -1):
            co[i] = (co[i] - co[i - 1]) / (xx[i] - xx[i - j])

    x = np.linspace(a, b, n_eval_pts)
    y = np.zeros_like(x, dtype=float)

    for c, xv in zip(co[::-1], xx[::-1]):
        y = y * (x - xv) + c

    return [x, y], [xx, yy] # DO NOT CHANGE

# CAUTION: The input arguments of the following method MUST NOT be changed.
# Any changes may cause the automated tests to fail and, hence, reduce
# the total score in the evaluation of your submission.
def newton_chebyshev(func: Callable[[float], float], bnds: list[float], n_eval_pts: int, n_sample_pts: int):
    """The method shall calculate the Newton interpolation of the function 'func' in the interval 'bnds'
       using 'n_sample_pts' Chebyshev sample points and 'n_eval_pts' evaluation points.
       
    Parameters
    ----------
    func : Callable
        Function that shall be approximated
    bnds : list[float]
        Interval in which 'func' shall be interpolated
    n_eval_pts : int
        Amount of evalutation points
    n_sample_pts : int
        Amount of sample points
        
    Returns
    -------
    [x, y] : list[np.ndarray]
        Evaluation points and values
    [xx, yy] : list[np.ndarray]
        Sample points and values
    """
    a, b = bnds
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)

    k = np.arange(n_sample_pts)
    xx = mid + half * np.cos((2 * k + 1) * np.pi / (2 * n_sample_pts))
    yy = func(xx)

    coeffs = yy.astype(float).copy()
    for j in range(1, n_sample_pts):
        for i in range(n_sample_pts - 1, j - 1, -1):
            coeffs[i] = (coeffs[i] - coeffs[i - 1]) / (xx[i] - xx[i - j])

    x = np.linspace(a, b, n_eval_pts)
    y = np.zeros_like(x, dtype=float)

    for c, xv in zip(coeffs[::-1], xx[::-1]):
        y = y * (x - xv) + c
        
    return [x, y], [xx, yy] # DO NOT CHANGE

# CAUTION: The input arguments of the following method MUST NOT be changed.
# Any changes may cause the automated tests to fail and, hence, reduce
# the total score in the evaluation of your submission.
def cubic_spline_equidistant(func: Callable[[float], float], bnds: list[float], n_eval_pts: int, n_sample_pts: int):
    """The method shall calculate the cubic spline interpolation of the function 'func' in the interval 'bnds'
        using 'n_sample_pts' equidistant sample points and 'n_eval_pts' evaluation points.
        
    Parameters
    ----------
    func : Callable
        Function that shall be approximated
    bnds : list[float]
        Interval in which 'func' shall be interpolated
    n_eval_pts : int
        Amount of evalutation points
    n_sample_pts : int
        Amount of sample points
        
    Returns
    -------
    [x, y] : list[np.ndarray]
        Evaluation points and values
    [xx, yy] : list[np.ndarray]
        Sample points and values
    """
    xx = []
    yy = []
    x  = []
    y  = []
    return [x, y], [xx, yy] # DO NOT CHANGE

# CAUTION: The input arguments of the following method MUST NOT be changed.
# Any changes may cause the automated tests to fail and, hence, reduce
# the total score in the evaluation of your submission.
def cubic_spline_chebyshev(func: Callable[[float], float], bnds: list[float], n_eval_pts: int, n_sample_pts: int):
    """The method shall calculate the cubic spline interpolation of the function 'func' in the interval 'bnds'
       using 'n_sample_pts' Chebyshev sample points and 'n_eval_pts' evaluation points.
       
    Parameters
    ----------
    func : Callable
        Function that shall be approximated
    bnds : list[float]
        Interval in which 'func' shall be interpolated
    n_eval_pts : int
        Amount of evalutation points
    n_sample_pts : int
        Amount of sample points
        
    Returns
    -------
    [x, y] : list[np.ndarray]
        Evaluation points and values
    [xx, yy] : list[np.ndarray]
        Sample points and values
    """
    xx = []
    yy = []
    x  = []
    y  = []
    return [x, y], [xx, yy] # DO NOT CHANGE
