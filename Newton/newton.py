# Automated check that you use the correct Python version #
from sys import version_info
if version_info[0] < 3 or version_info[1] < 10:
    raise Exception("Must be using Python 3.10 or newer")
###########################################################

import numpy as np
from typing import Callable

def newton(f: Callable[[float], float], df: Callable[[float], float], x0: float, tol: float = 1e-8,
           maxiter: int = 50) -> np.ndarray[float]:
    """
        This method shall implement the local Newton method,
        that calculates a zero of some given function f
    Input:
        f : Callable  -> Function of which to find a zero
        df : Callable -> Derivative of f
        x0 : float    -> Initial point
        tol: float    -> Tolerance for the stop criterion (optional)
        maxiter : int -> Maximal number of iterations
    Output:
        np.ndarray -> Array of the iterates x^(0), x^(1),..., x^(N)
    """
    xs = [x0]
    xk = x0

    for _ in range(maxiter):
        fx = f(xk)
        dfx = df(xk)

        if abs(fx) < tol:
            break

        if abs(dfx) < 1e-14:
            raise ValueError(f"Derivative is zero at x = {xk}. Cannot proceed.")

        xk = xk - fx / dfx
        xs.append(xk)

    else:
        if abs(f(xk)) >= tol:
            raise ValueError("Newton method did not converge within maxiter.")

    return np.array(xs)

def newton_global(f: Callable[[float], float], df: Callable[[float], float], x0: float, tol: float = 1e-8,
                  maxiter: int = 50, beta: float = 0.5, delta: float = 1e-3) -> np.ndarray[float]:
    """
        This method shall implement the global Newton method,
        that calculates a zero of some given function f.
        It uses the Armijo search to determine the appropriate
        step width in each iteration.
    Input:
        f : Callable  -> Function of which to find a zero
        df : Callable -> Derivative of f
        x0 : float    -> Initial point
        tol: float    -> Tolerance for the stop criterion (optional)
        maxiter : int -> Maximal number of iterations
        beta: float   -> Reduction parameter for the step width σ_k (σ_k = β*σ_{k-1} if needed, optional)
        delta: float  -> The Armijo constant (optional)
    Output:
        np.ndarray -> Array of the iterates x^(0), x^(1),..., x^(N)
    """
    xs = [x0]
    xk = x0

    for _ in range(maxiter):
        fx = f(xk)
        dfx = df(xk)

        if abs(fx) < tol:
            break

        if abs(dfx) < 1e-14:
            raise ValueError(f"Derivative is zero at x = {xk}. Cannot proceed.")

        sk = -fx / dfx
        sigma = 1.0

        while sigma >= 1e-12:
            x_trial = xk + sigma * sk
            if abs(f(x_trial)) <= (1 - delta * sigma) * abs(fx):
                break
            sigma *= beta
        else:
            raise RuntimeError("Armijo line search failed. Step size too small.")

        xk = xk + sigma * sk
        xs.append(xk)

    else:
        if abs(f(xk)) >= tol:
            raise ValueError("Global Newton method did not converge within maxiter.")

    return np.array(xs)