# Automated check that you use the correct Python version
from sys import version_info

if version_info[0] < 3 or version_info[1] < 10:
    raise Exception("Must be using Python 3.10 or newer")
###########################################################
import numpy as np
import matplotlib.pyplot as plt
from lgs import check_input, gauss, frobenius_norm

if __name__ == '__main__':
# 4.2a
    A1 = np.array([[3, 1, 0, 0, 0],[1, 3, 1, 0, 0],[0, 1, 3, 1, 0],[0, 0, 1, 3, 1],[0, 0, 0, 1, 3]], float)
    b1 = np.array([4, 5, 5, 5, 4], float)
    x1 = gauss(A1, b1)
    print("Solution for 4.2a:", x1)

    #4.2b
    A2 = np.array([[1, 2, 2],[2, 4, 6],[1,-1, 1]], float)
    b2 = np.array([2, 5, -3], float)
    try:
        x2 = gauss(A2, b2)
        print("Solution for 4.2b:", x2)
    except Exception as e:
        print("Error solving 4.2b:", e)

    #4.2c 
    t_vals      = list(range(10))
    conds       = []
    rel_errors  = []
    b           = np.array([0, 1], float)
    b_perturb   = np.array([1e-5, 0], float)

    for t in t_vals:
        A = np.array([[2, 1],
                      [1, 2 * 10**t]], float)
        normA   = frobenius_norm(A)

        det = 4 * 10**t - 1
        invA = np.array([[ 2 * 10**t, -1], [ -1,  2]], float) / det

        normInvA = frobenius_norm(invA)

        cond = normA * normInvA
        conds.append(cond)

        x      = gauss(A, b)
        x_tilde = gauss(A, b + b_perturb)

        rel_err = np.linalg.norm(x_tilde - x) / np.linalg.norm(x)
        rel_errors.append(rel_err)

    plt.figure()
    plt.loglog(conds, rel_errors, marker='o', label='Relative error')
    plt.xlabel('Condition number ‖A‖ₙ · ‖A⁻¹‖ₙ')
    plt.ylabel('Relative error ‖x̃–x‖₂/‖x‖₂')
    plt.title('Relative Error vs. Condition Number')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.savefig('Rel_Error_VS_Cond.pdf')
    plt.show()