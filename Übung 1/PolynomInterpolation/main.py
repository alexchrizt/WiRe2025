# Automated check that you use the correct Python version
from sys import version_info
if version_info[0] < 3 or version_info[1] < 10:
    raise Exception("Must be using Python 3.10 or newer")
###########################################################
from scipy.interpolate import CubicSpline
from interpolation import newton_equidistant, newton_chebyshev, cubic_spline_equidistant, cubic_spline_chebyshev
import numpy as np
import matplotlib.pyplot as plt

# Ziel-Funktion definieren
func = lambda x: 1 / (1 + x**2)

# Intervall und Auswertungspunkte
bnds = [-5, 5]
n_eval_pts = 1000
n_sample_pts = 7
x_eval = np.linspace(bnds[0], bnds[1], n_eval_pts)
f_true = func(x_eval)

# Newton Equidistant
(x1, y1), (xx1, yy1) = newton_equidistant(func, bnds, n_eval_pts, n_sample_pts)
plt.plot(x_eval, f_true, 'k-', label=r'$f(x) = \frac{1}{1 + x^2}$')
plt.plot(x1, y1, 'r--', label='Interpolierte Funktion')
plt.plot(xx1, yy1, 'bx', label='Stützstellen')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Interpolation mit Newton (äquidistant)')
plt.legend()
plt.grid()
plt.savefig('Plots/Interpolation_Newton_equidistant.pdf', dpi=300)
plt.show()
err1 = np.abs(f_true - y1) / np.abs(f_true)
err1 = np.clip(err1, 1e-16, 1e10)

# Newton Chebyshev
(x2, y2), (xx2, yy2) = newton_chebyshev(func, bnds, n_eval_pts, n_sample_pts)
plt.plot(x_eval, f_true, 'k-', label=r'$f(x) = \frac{1}{1 + x^2}$')
plt.plot(x2, y2, 'r--', label='Interpolierte Funktion')
plt.plot(xx2, yy2, 'bx', label='Stützstellen')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Interpolation mit Newton (Chebyshev)')
plt.legend()
plt.grid()
plt.savefig('Plots/Interpolation_Newton_Chebyshev.pdf', dpi=300)
plt.show()
err2 = np.abs(f_true - y2) / np.abs(f_true)
err2 = np.clip(err2, 1e-16, 1e10)

# Cubic Spline Equidistant
(x3, y3), (xx3, yy3) = cubic_spline_equidistant(func, bnds, n_eval_pts, n_sample_pts)
plt.plot(x_eval, f_true, 'k-', label=r'$f(x) = \frac{1}{1 + x^2}$')
plt.plot(x3, y3, 'r--', label='Interpolierte Funktion')
plt.plot(xx3, yy3, 'bx', label='Stützstellen')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Interpolation mit kubischem Spline (äquidistant)')
plt.legend()
plt.grid()
plt.savefig('Plots/Interpolation_Cubic_Spline_equidistant.pdf', dpi=300)
plt.show()
err3 = np.abs(f_true - y3) / np.abs(f_true)
err3 = np.clip(err3, 1e-16, 1e10)

# Cubic Spline Chebyshev
(x4, y4), (xx4, yy4) = cubic_spline_chebyshev(func, bnds, n_eval_pts, n_sample_pts)
plt.plot(x_eval, f_true, 'k-', label=r'$f(x) = \frac{1}{1 + x^2}$')
plt.plot(x4, y4, 'r--', label='Interpolierte Funktion')
plt.plot(xx4, yy4, 'bx', label='Stützstellen')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Interpolation mit kubischem Spline (Chebyshev)')
plt.legend()
plt.grid()
plt.savefig('Plots/Interpolation_Cubic_Spline_Chebyshev.pdf', dpi=300)
plt.show()
err4 = np.abs(f_true - y4) / np.abs(f_true)
err4 = np.clip(err4, 1e-16, 1e10)

# Vergleich: relative Fehler in semilogy
plt.semilogy(x_eval, err1, label='Newton äquidistant')
plt.semilogy(x_eval, err2, label='Newton Chebyshev')
plt.semilogy(x_eval, err3, label='Spline äquidistant')
plt.semilogy(x_eval, err4, label='Spline Chebyshev')
plt.xlabel(r'$x$')
plt.ylabel(r'Relativer Fehler')
plt.title('Relative Fehler der Interpolationen')
plt.legend()
plt.grid()
plt.savefig('Plots/Interpolation_Rel_Errors.pdf', dpi=300)
plt.show()