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
    xx = []
    yy = []
    x  = []
    y  = []
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
    xx = []
    yy = []
    x  = []
    y  = []
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
