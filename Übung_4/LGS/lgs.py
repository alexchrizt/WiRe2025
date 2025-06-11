# Automated check that you use the correct Python version #
from sys import version_info
if version_info[0] < 3 or version_info[1] < 10:
    raise Exception("Must be using Python 3.10 or newer")
###########################################################

import numpy as np
import warnings

def check_input(A: np.ndarray[float], b: np.ndarray[float]) -> None:
    """
        This method shall check the input A & b and throw an error or a warning if necessary
    Input:
        A: np.ndarray -> System matrix (is checked for symmetry and diagonal dominance)
        b: np.ndarray -> Right-hand side vector (is checked for the correct length)
    Output:
        None
    """
    # Check: A is square
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")

    # Check: b has correct length
    if A.shape[0] != b.shape[0]:
        raise ValueError("Vector b must have same length as the number of rows in A.")

    # Check: strict diagonal dominance
    n = A.shape[0]
    for i in range(n):
        diagonal = abs(A[i, i])
        off_diagonal_sum = np.sum(np.abs(A[i, :])) - diagonal
        if diagonal <= off_diagonal_sum:
            warnings.warn(f"Matrix A is not strictly diagonally dominant at row {i}.", UserWarning)
            break  # One warning is enough


def gauss(A: np.ndarray[float], b: np.ndarray[float], tol: float = 1e-12) -> np.ndarray[float]:
    """
        This method shall solve the linear system Ax = b using the Gauß algorithm without pivot search
    Input:
        A: np.ndarray -> System matrix
        b: np.ndarray -> Right-hand side vector
        c: float      -> Tolerance for the stop criterion (optional argument)
    Output:
        x: np.ndarray -> Solution vector of the system
    """
    check_input(A, b)

    A = A.copy().astype(float)
    b = b.copy().astype(float)
    n = A.shape[0]

    # Forward elimination
    for i in range(n):
        if abs(A[i, i]) <= tol:
            raise ValueError(f"Pivot too small at row {i}: |A[{i},{i}]| <= {tol}")
        
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - factor * A[i, i:]
            b[j] = b[j] - factor * b[i]

    # Backward substitution
    x = np.zeros(n)

    for i in reversed(range(n)):
        if abs(A[i, i]) <= tol:
            raise ValueError(f"Zero pivot encountered at row {i} during back substitution.")
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    #TODO Do not forget to guarantee that tol is an optional input argument with standard value 1e-12
    return x


def frobenius_norm(A: np.ndarray[float]) -> float:
    """
        This method shall compute the Frobenius norm of the matrix A
    Input:
        A: np.ndarray -> System matrix whose Frobenius norm shall be calculated
    Output:
        float -> Frobenius norm ||A||_F of A
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square to compute the Frobenius norm.")

    return np.sqrt(np.sum(A ** 2))