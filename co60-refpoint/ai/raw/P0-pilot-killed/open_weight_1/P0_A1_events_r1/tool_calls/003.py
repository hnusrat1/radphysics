import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.integrate import quad
import random

# Constants
rho = 1.0  # g/cm3
me = 0.511  # electron rest mass energy MeV

# Data from NIST XCOM for water (mass attenuation coefficients cm2/g)
# Energies in MeV
energies = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1,
                     0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0,
                     3.0, 4.0, 5.0, 6.0, 8.0, 10.0])

# Coherent (Rayleigh)
mu_coh = np.array([1.77e-1, 1.75e-1, 1.66e-1, 1.60e-1, 1.50e-1, 1.38e-1,
                   1.26e-1, 1.06e-1, 1.03e-1, 8.47e-2, 6.72e-2, 4.56e-2,
                   3.30e-2, 2.58e-2, 2.07e-2, 1.35e-2, 9.21e-3, 4.86e-3,
                   2.87e-3, 1.39e-3, 8.8e-4, 6.1e-4, 4.5e-4, 2.8e-4, 2.0e-4])

# Incoherent (Compton)
mu_inc = np.array([1.72e-1, 1.70e-1, 1.62e-1, 1.56e-1, 1.49e-1, 1.35e-1,
                   1.27e-1, 1.17e-1, 1.12e-1, 1.03e-1, 1.29e-1, 1.19e-1,
                   1.11e-1, 1.01e-1, 9.32e-2, 8.24e-2, 7.07e-2, 5.80e-2,
                   4.93e-2, 3.91e-2, 3.15e-2, 2.58e-2, 2.10e-2, 1.44e-2, 1.02e-2])

# Photoelectric
mu_ph = np.array([4.78e3, 1.56e3, 1.05e3, 4.88e2, 2.56e2, 1.31e2, 7.28e1,
                  3.06e1, 1.76e1, 5.11e0, 2.20e0, 5.38e-1, 2.61e-1, 1.06e-1,
                  6.48e-2, 4.75e-2, 3.50e-2, 2.35e-2, 1.82e-2, 1.39e-2,
                  1.16e-2, 1.01e-2, 9.15e-3, 7.60e-3, 6.40e-3])

# Pair production (nuclear + triplet) 
mu_pair = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.00e-3, 1.50e-3,
                    1.77e-3, 3.45e-3, 5.17e-3, 6.96e-3, 8.35e-3, 1.06e-2, 1.18e-2])

# Actually for pair, energies below threshold (1.022 MeV) are zero.
# We'll set them to zero but our interpolation may cause issues; we'll handle.

# Create interpolation functions log-log
# We'll clip energies below 0.01 and above 10
def interp_log(x, y):
    # Interpolate log-log, but handle zeros
    # For mu_pair, some values are zero; we set log(0) = -inf, but we can treat below threshold as 0.
    ok = y > 0
    x_ok = x[ok]
    y_ok = y[ok]
    log_x = np.log(x_ok)
    log_y = np.log(y_ok)
    f = interp1d(log_x, log_y, kind='linear', fill_value='extrapolate')
    def fun(xi):
        if xi < x_ok[0]:
            # extrapolate below lowest energy (but we may not need)
            return np.exp(f(np.log(x_ok[0])))  # constant? better to extrapolate
        if xi > x_ok[-1]:
            return np.exp(f(np.log(x_ok[-1])))
        return np.exp(f(np.log(xi)))
    return fun

# For coherent, we use interpolation
mu_coh_func = interp_log(energies, mu_coh)
mu_inc_func = interp_log(energies, mu_inc)
mu_ph_func = interp_log(energies, mu_ph)

# For pair, we need to handle threshold
def mu_pair_func(x):
    if x <= 1.022: # 2*me
        return 0.0
    # interpolate from energies above threshold
    # use data from 1.5 upwards
    idx = energies >= 1.5
    return interp_log(energies[idx], mu_pair[idx])(x)

# Test
print(mu_coh_func(2.0), mu_inc_func(2.0), mu_ph_func(2.0), mu_pair_func(2.0))