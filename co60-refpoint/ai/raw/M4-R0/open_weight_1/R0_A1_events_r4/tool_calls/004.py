import numpy as np
import math

mec2 = 0.511
r0 = 2.8179403262e-13
n_e_per_cm3 = 3.3456e23

def mu_compton(E):
    if E <= 0:
        return 0.0
    a = E / mec2
    a2 = a*a
    x = 1.0 + 2.0*a
    lnx = math.log(x)
    term1 = (1.0+a)/a2 * (2.0*(1.0+a)/x - lnx/a)
    term2 = lnx/(2.0*a) - (1.0+3.0*a)/(x*x)
    sigma = 2.0*math.pi*r0*r0 * (term1 + term2)
    return sigma * n_e_per_cm3

def mu_pair(E):
    if E <= 2*mec2:
        return 0.0
    e_vals = np.array([1.1732, 1.3325, 2.0, 3.0, 5.0, 10.0])
    mu_vals = np.array([0.006205, 0.006286, 0.0082, 0.0109, 0.0126, 0.0120])
    if E <= 1.1732:
        t = E - 2*mec2
        if t <= 0:
            return 0.0
        t0 = 1.1732 - 1.022
        mu0 = 0.006205
        return mu0 * (t/t0)**2 * 1.0
    elif E >= 10.0:
        return 0.0120
    else:
        logE = math.log(E)
        logE_vals = np.log(e_vals)
        logmu_vals = np.log(mu_vals)
        return math.exp(np.interp(logE, logE_vals, logmu_vals)) * 1.0

def mu_pe(E):
    if E <= 0:
        return 0.0
    e_vals = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.1732, 1.3325, 2.0])
    mu_vals = np.array([5.33, 0.742, 0.0321, 0.0164, 0.0097, 0.0057, 0.0026, 0.0011, 0.001735, 0.001524, 0.00078])
    if E <= 0.01:
        return mu_vals[0] * (E/0.01)**(-3.5) * 1.0
    elif E >= 2.0:
        return 0.00078 * (E/2.0)**(-3.0) * 1.0
    else:
        logE = math.log(E)
        logE_vals = np.log(e_vals)
        logmu_vals = np.log(mu_vals)
        return math.exp(np.interp(logE, logE_vals, logmu_vals)) * 1.0

def mu_rayleigh(E):
    if E <= 0:
        return 0.0
    e_vals = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.1732, 1.3325, 2.0, 5.0])
    mu_vals = np.array([0.0008, 0.0010, 0.0012, 0.0016, 0.0021, 0.0024, 0.0025, 0.00253, 0.00213, 0.0012, 0.0003])
    if E <= 0.01:
        return 0.0008 * (E/0.01)**0.3 * 1.0
    elif E >= 5.0:
        return 0.0003 * (E/5.0)**(-1.5) * 1.0
    else:
        logE = math.log(E)
        logE_vals = np.log(e_vals)
        logmu_vals = np.log(mu_vals)
        return math.exp(np.interp(logE, logE_vals, logmu_vals)) * 1.0

# Test
for E in [1.1732, 1.3325, 0.5, 0.2, 0.1]:
    mu_c = mu_compton(E)
    mu_p = mu_pair(E)
    mu_pe_val = mu_pe(E)
    mu_r = mu_rayleigh(E)
    mu_t = mu_c + mu_p + mu_pe_val + mu_r
    print(f"E={E:.4f} MeV: total={mu_t:.6f}, compton={mu_c:.6f}, pair={mu_p:.6f}, pe={mu_pe_val:.6f}, rayleigh={mu_r:.6f}")