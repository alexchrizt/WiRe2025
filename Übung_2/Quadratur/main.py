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
    plt.title('Local Newton-Cotes: $f(x) = \cos(x) + e^x + 3x^2$ on $[0, 5]$')
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
    plt.title('Monte-Carlo: $f(x) = \cos(x) + e^x + 3x^2$ on $[0, 5]$')
    plt.legend()
    plt.grid(True)
    mc_path = os.path.join(plots_dir, "Monte_Carlo.pdf")
    plt.savefig(mc_path)
    print(f"Monte-Carlo plot saved: {mc_path}")
    plt.close()

    # Local Newton-Cotes Convergence Plot
    h_vals = 5 / (2 ** np.arange(10))
    errors_nc_n1 = [abs(newton_cotes_quadrature(f, 0, h, 1) - (np.sin(h) + np.exp(h) + h**3)) for h in h_vals]
    errors_nc_n2 = [abs(newton_cotes_quadrature(f, 0, h, 2) - (np.sin(h) + np.exp(h) + h**3)) for h in h_vals]

    plt.figure()
    plt.loglog(h_vals, errors_nc_n1, marker='o', linestyle='', label='Trapezoidal (n=1)')
    plt.loglog(h_vals, errors_nc_n2, marker='s', linestyle='', label="Simpson's (n=2)")
    plt.loglog(h_vals, h_vals**2 * errors_nc_n1[0]/h_vals[0]**2, 'k--', label='O(h^2)')
    plt.loglog(h_vals, h_vals**4 * errors_nc_n2[0]/h_vals[0]**4, 'k-.', label='O(h^4)')
    plt.xlabel('Interval Length h (log scale)')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('Local Newton-Cotes Error (log-log)')
    plt.grid(True, which='both')
    plt.legend()
    local_path = os.path.join(plots_dir, "Local_Convergence_Plot.pdf")
    plt.savefig(local_path)
    print(f"Local Newton-Cotes plot saved: {local_path}")
    plt.close()

    # Global Newton-Cotes Convergence Plot
    m_values = np.arange(1, 101)
    h_vals_global = 5 / m_values
    errors_global_n1 = [abs(global_newton_cotes_quadrature(f, a, b, 1, m) - true_val) for m in m_values]
    errors_global_n2 = [abs(global_newton_cotes_quadrature(f, a, b, 2, m) - true_val) for m in m_values]

    plt.figure()
    plt.loglog(h_vals_global, errors_global_n1, marker='o', linestyle='', label='Trapezoidal (n=1)')
    plt.loglog(h_vals_global, errors_global_n2, marker='s', linestyle='', label="Simpson's (n=2)")
    plt.loglog(h_vals_global, h_vals_global**2 * errors_global_n1[0]/h_vals_global[0]**2, 'k--', label='O(h^2)')
    plt.loglog(h_vals_global, h_vals_global**4 * errors_global_n2[0]/h_vals_global[0]**4, 'k-.', label='O(h^4)')
    plt.xlabel('Interval Length h (log scale)')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('Global Newton-Cotes Error (log-log)')
    plt.grid(True, which='both')
    plt.legend()
    global_path = os.path.join(plots_dir, "Global_Convergence_Plot.pdf")
    plt.savefig(global_path)
    print(f"Global Newton-Cotes plot saved: {global_path}")
    plt.close()

    # Monte-Carlo Convergence Plot
    np.random.seed(0)
    mc_samples = np.logspace(1, 4, num=50, dtype=int)
    vals_mc = [monte_carlo_quadrature(f, a, b, n) for n in mc_samples]
    errors_mc = [abs(val - true_val) for val in vals_mc]

    plt.figure()
    plt.loglog(mc_samples, errors_mc, linestyle='', marker='o', alpha=0.7, label='Monte-Carlo')
    plt.loglog(mc_samples, (1/np.sqrt(mc_samples)) * errors_mc[0]*np.sqrt(mc_samples[0]), 'k--', label='O(1/sqrt(n))')
    plt.xlabel('Samples n (log scale)')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('Monte-Carlo Integration Error (log-log)')
    plt.grid(True, which='both')
    plt.legend()
    mc_conv_path = os.path.join(plots_dir, "Monte_Carlo_Convergence_Plot.pdf")
    plt.savefig(mc_conv_path)
    print(f"Monte-Carlo convergence plot saved: {mc_conv_path}")
    plt.close()
