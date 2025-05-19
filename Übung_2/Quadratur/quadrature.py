import numpy as np
from scipy import integrate
from typing import Callable

# CAUTION: The input arguments of the following method MUST NOT be changed.
# Any changes may cause the automated tests to fail and, hence, reduce
# the total score in the evaluation of your submission.
def newton_cotes_weights(n: int) -> np.ndarray[float]:
    """This method calculates the Newton-Cotes weights for an interval [0, 1]
       with n+1 equidistant points.
        
    Parameters
    ----------
    n : int
        Degree of the interpolation polynomial (n+1 points)
        
    Returns
    -------
    np.ndarray
        Array of Newton-Cotes weights.
    """
    weights = np.zeros(n + 1)
    nodes = np.linspace(0, 1, n + 1)

    # integrate every L basis polynomial -> weights
    for k in range(n + 1):  
        current_node = nodes[k]
        denominator = 1.0
        for j, node_j in enumerate(nodes):
            if j != k:
                denominator *= (current_node - node_j)

        def L_k_integrate(x: float) -> float: # assume denom non zero, calc the k-th Lagrange polynomial
            numerator = 1.0
            for j, node_j in enumerate(nodes):
                if j != k:
                    numerator *= (x - node_j)
            return numerator / denominator

        weights[k], _ = integrate.quad(L_k_integrate, 0, 1) # quad returns: (value, error)
    
    return weights # DO NOT CHANGE

# CAUTION: The input arguments of the following method MUST NOT be changed.
# Any changes may cause the automated tests to fail and, hence, reduce
# the total score in the evaluation of your submission.
def newton_cotes_quadrature(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """This method shall calculate the integral o'f' from 'a' to 'b' using
       the Newton-Cotes rule.
        
    Parameters
    ----------
    f : Callable
        The function to be integrated
    a : float
        Lower bound of the integration interval
    b : float
        Upper bound of the integration interval
    n : int
        Degree of the interpolation polynomial
    
    Returns
    -------
    float
        Approximated value of the integral
    """
    integral = 0.0
    w = newton_cotes_weights(n)
    nodes = np.linspace(a, b, n + 1)
    
    integral_sum = 0.0
    for i in range(n + 1):
        integral_sum += w[i] * f(nodes[i])
            
    integral = (b - a) * integral_sum
    return integral # DO NOT CHANGE

# CAUTION: The input arguments of the following method MUST NOT be changed.
# Any changes may cause the automated tests to fail and, hence, reduce
# the total score in the evaluation of your submission.
def global_newton_cotes_quadrature(f: Callable[[float], float], a: float, b: float, n: int, m: int) -> float:
    """This method shall calculate the integral o'f' from 'a' to 'b' using
       the global Newton-Cotes rule.
    
    Parameters
    ----------
    f : Callable
        The function to be integrated
    a : float
        Lower bound of the integration interval 
    b : float
        Upper bound of the integration interval
    n : int
        Degree of the interpolation polynomial
    m : int
        Amount of subintervals
    
    Returns
    -------
    float
        Approximated value of the integral
    """
    integral = 0.0
    subinterval_nodes = np.linspace(a, b, m + 1) # assuming: m > 0
    for i in range(m):
        integral+= newton_cotes_quadrature(f, subinterval_nodes[i], subinterval_nodes[i+1], n)
        
    return integral # DO NOT CHANGE

# CAUTION: The input arguments of the following method MUST NOT be changed.
# Any changes may cause the automated tests to fail and, hence, reduce
# the total score in the evaluation of your submission.
def monte_carlo_quadrature(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """This method shall calculate the integral o'f' from 'a' to 'b' using
       a Monte-Carlo approach.
    
    Parameters
    ----------
    f : Callable
        The function to be integrated
    a : float
        Lower bound of the integration interval
    b : float
        Upper bound of the integration interval
    n : int
        Number of samples
    
    Returns
    -------
    float
        Approximated value of the integral
    """
    integral = 0.0
    r = np.random.uniform(a, b, n)
    sum_f = 0.0
    for i in r:
        sum_f += f(i)
        
    integral = (b - a) * (sum_f / n)
    return integral # DO NOT CHANGE
