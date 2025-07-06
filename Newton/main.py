# Automated check that you use the correct Python version
from sys import version_info

if version_info[0] < 3 or version_info[1] < 10:
    raise Exception("Must be using Python 3.10 or newer")
###########################################################
import numpy as np
import matplotlib.pyplot as plt
from newton import newton, newton_global

if __name__ == '__main__':
    # 5.2a
    f1 = lambda x: (1/20) * x**3 + x - 2 + np.cos((6/5) * x)
    df1 = lambda x: (3/20) * x**2 + 1 - (6/5) * np.sin((6/5) * x)
    x0_1 = 3.5
    xs1 = newton(f1, df1, x0_1, maxiter=30)
    root1 = 2.31448778999

    xs_plot = np.linspace(2, 4, 400)
    plt.figure()
    plt.plot(xs_plot, f1(xs_plot), label='f(x)')
    plt.axhline(0, color='k', linewidth=0.8)
    plt.axvline(root1, color='k', linestyle='--', linewidth=0.8, label='Root')
    for k, xk in enumerate(xs1):
        plt.plot(xk, f1(xk), marker='x', markersize=8)
        plt.annotate(str(k), (xk, f1(xk)), textcoords='offset points', xytext=(5,5))

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Local Newton Method: Example 1')
    plt.legend()
    plt.savefig('Local_Newton_Ex_1.pdf')
    plt.close()

     # 5.2b
    f2  = lambda x: x**3 - 2*x + 2
    df2 = lambda x: 3*x**2 - 2
    x0   = 0.0
    β    = 0.33
    δ    = 1e-3

    try:
        xs_local = newton(f2, df2, x0, tol=1e-8, maxiter=50)
    except ValueError as e:
        print("Local Newton didn’t converge:", e)
        xs_local = []           

    xs_global = newton_global(f2, df2, x0, tol=1e-8,
                              maxiter=50, beta=β, delta=δ)
    root = xs_global[-1]

    xs_plot = np.linspace(-2, 2, 400)
    plt.figure()
    plt.plot(xs_plot, f2(xs_plot), label='f(x)')
    plt.axhline(0, color='k', linewidth=0.8)
    plt.axvline(root, color='k', linestyle='--', linewidth=0.8,
                label=f'root ≈ {root:.4f}')

    for k, xk in enumerate(xs_local):
        plt.plot(xk, f2(xk), 'x', markersize=8, label='Local' if k==0 else '')
        plt.annotate(str(k), (xk, f2(xk)), textcoords='offset points', xytext=(5,5))

    for k, xk in enumerate(xs_global):
        plt.plot(xk, f2(xk), 'o', markersize=6, label='Global' if k==0 else '')
        plt.annotate(str(k), (xk, f2(xk)), textcoords='offset points', xytext=(5,-15))

    plt.xlim(-2,2)
    plt.ylim(min(f2(xs_plot)), max(f2(xs_plot)))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Local vs Global Newton Method')
    plt.legend()
    plt.savefig('Local_vs_Global.pdf')
    plt.close()

    plt.figure()
    if len(xs_local) > 0:
        errs_local = np.abs(xs_local - root)
        plt.plot(range(len(errs_local)), errs_local, 'x-', label='Local')
    errs_global = np.abs(xs_global - root)
    plt.plot(range(len(errs_global)), errs_global, 'o-', label='Global')

    plt.yscale('log')
    plt.xlabel('Iteration k')
    plt.ylabel(r'$|x^{(k)} - x^*|$')
    plt.title('Error vs Iteration')
    plt.legend()
    plt.savefig('Local_vs_Global_Error.pdf')
    plt.close()
    
    # 5.2c)
    plt.figure()
    for n in [2, 3, 4]:
        f3 = lambda x, n=n: x**n - 2
        df3 = lambda x, n=n: n * x**(n-1)
        xs3 = newton(f3, df3, 0.5)
        root3 = 2**(1/n)
        errors = np.abs(xs3 - root3)
        plt.semilogy(range(len(errors)), errors, marker='o', label=f'n={n}')
    plt.xlabel('Iteration k')
    plt.ylabel('|x(k) - x*|')
    plt.title('Convergence Plot for Local Newton')
    plt.legend()
    plt.savefig('Convergence_Plot.pdf')
    plt.close()
