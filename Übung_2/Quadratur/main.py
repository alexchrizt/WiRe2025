# Automated check that you use the correct Python version
from sys import version_info

if version_info[0] < 3 or version_info[1] < 10:
    raise Exception("Must be using Python 3.10 or newer")
###########################################################
from scipy.interpolate import CubicSpline
from quadrature import newton_cotes_quadrature, global_newton_cotes_quadrature, monte_carlo_quadrature
import os
import numpy as np
import matplotlib.pyplot as plt

def f(x: float) -> float:
    return np.cos(x) + np.exp(x) + 3 * x**2

if __name__ == '__main__':
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    a, b = 0.0, 5.0 
    true_val = (np.sin(b) + np.exp(b) + b**3) - (np.sin(a) + np.exp(a) + a**3)
    print(f"Analytical integral value: {true_val}")

    n_nc_orders = np.arange(1, 21)
    vals_nc = [newton_cotes_quadrature(f, a, b, n_val) for n_val in n_nc_orders]

    plt.figure()
    plt.plot(n_nc_orders, vals_nc, marker='o', linestyle='-', label='Newton-Cotes Approx.')
    plt.axhline(y=true_val, color='gray', linestyle='--', label=f'Analytical ({true_val:.4f})')
    plt.xlabel('Order (n)')
    plt.ylabel('Integral Value')
    plt.title('Local Newton-Cotes: $f(x) = \\cos(x) + e^x + 3x^2$ on $[0, 5]$')
    plt.legend()
    plt.grid(True)
    plt.xticks(n_nc_orders)
    nc_path = os.path.join(plots_dir, "Newton_Cotes.pdf")
    plt.savefig(nc_path)
    print(f"Newton-Cotes plot saved: {nc_path}")
    plt.close()

    n_mc_samples = np.arange(1, 10001)
    vals_mc = [monte_carlo_quadrature(f, a, b, n_val) for n_val in n_mc_samples]

    plt.figure()
    plt.plot(n_mc_samples, vals_mc, linestyle='-', alpha=0.7, label='Monte-Carlo Approx.')
    plt.axhline(y=true_val, color='gray', linestyle='--', label=f'Analytical ({true_val:.4f})')
    plt.xlabel('Samples (n)')
    plt.ylabel('Integral Value')
    plt.title('Monte-Carlo: $f(x) = \\cos(x) + e^x + 3x^2$ on $[0, 5]$')
    plt.legend()
    plt.grid(True)
    mc_path = os.path.join(plots_dir, "Monte_Carlo.pdf")
    plt.savefig(mc_path)
    print(f"Monte-Carlo plot saved: {mc_path}")
    plt.close()

    #print("\nFill in beobachtungen.txt.")