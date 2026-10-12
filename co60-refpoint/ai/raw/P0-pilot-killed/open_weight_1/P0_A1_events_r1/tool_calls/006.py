import numpy as np
import math
from scipy.interpolate import interp1d

# Correct NIST XCOM data for water (mass attenuation coefficients cm2/g)
energies = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
mu_coh = np.array([0.03094, 0.01858, 0.01234, 0.006699, 0.003458, 0.002573, 0.001571, 0.001105, 0.00063, 0.00039, 0.00027, 0.00019, 0.00012, 0.00008])
mu_inc = np.array([0.1237, 0.1153, 0.1042, 0.08908, 0.07439, 0.06756, 0.05565, 0.04748, 0.03454, 0.02668, 0.02087, 0.01681, 0.01162, 0.00883])
mu_ph = np.array([0.00627, 0.00260, 0.00125, 0.00047, 0.00014, 0.000089, 0.000029, 0.000013, 0.000004, 0.0000017, 8e-7, 0.0, 0.0, 0.0])
mu_pair = np.array([0.0, 0.0, 0.0, 0.0, 0.00015, 0.00046, 0.00150, 0.00177, 0.00345, 0.00517, 0.00696, 0.00835, 0.0106, 0.0118])

# Create interpolation functions log-log
def create_interp(x, y):
    # Remove zeros for log
    mask = y > 0
    if np.any(mask):
        x_ok = x[mask]
        y_ok = y[mask]
        logx = np.log(x_ok)
        logy = np.log(y_ok)
        f = interp1d(logx, logy, kind='linear', fill_value='extrapolate')
        def fun(xi):
            if xi <= x_ok[0]:
                return y_ok[0]  # constant below lowest? Actually extrapolate
            if xi >= x_ok[-1]:
                return y_ok[-1]
            return np.exp(f(np.log(xi)))
        return fun
    else:
        return lambda xi: 0.0

mu_coh_func = create_interp(energies, mu_coh)
mu_inc_func = create_interp(energies, mu_inc)
mu_ph_func = create_interp(energies, mu_ph)
mu_pair_func = create_interp(energies, mu_pair)

# Test at 2 MeV
print("At 2 MeV: coh={:.6f} inc={:.6f} ph={:.6f} pair={:.6f}".format(
    mu_coh_func(2.0), mu_inc_func(2.0), mu_ph_func(2.0), mu_pair_func(2.0)))

# Test at 0.5 MeV
print("At 0.5 MeV: coh={:.6f} inc={:.6f} ph={:.6f} pair={:.6f}".format(
    mu_coh_func(0.5), mu_inc_func(0.5), mu_ph_func(0.5), mu_pair_func(0.5)))