import numpy as np
import time

# Constants
r_e = 2.8179403227e-13  # cm
me = 0.5109989461  # MeV
NA = 6.02214076e23
M_H2O = 18.015
rho = 1.0
N_e = 10

E1 = 1.1732
E2 = 1.3325

# Phantom bounds
XMIN, XMAX = -15.0, 15.0
YMIN, YMAX = -15.0, 15.0
ZMIN, ZMAX = 0.0, 30.0

# Scoring volume
SV_R = 1.0
SV_ZLO = 4.9
SV_ZHI = 5.1

def sigma_KN(E):
    k = E / me
    t1 = (1+k)/(k*k)
    t2 = 2*(1+k)/(1+2*k) - np.log(1+2*k)/k
    t3 = np.log(1+2*k)/(2*k)
    t4 = (1+3*k)/((1+2*k)*(1+2*k))
    term = t1*t2 + t3 - t4
    return 2*np.pi * r_e**2 * term

def sigma_total_approx(E):
    sig_comp = N_e * sigma_KN(E)
    sig_rayl = 0.0014 * M_H2O / NA * (1.0 / max(E, 0.01))**2
    sig_pe = 0.0016 * M_H2O / NA * (1.0 / max(E, 0.01))**3.5
    if E > 1.022:
        sig_pair = 0.00087 * M_H2O / NA * (E - 1.022) / (1.25 - 1.022)
    else:
        sig_pair = 0.0
    total = sig_comp + sig_rayl + sig_pe + sig_pair
    return total, sig_comp, sig_rayl, sig_pe, sig_pair

# Oops too many errors. Let me write a clean version from scratch.