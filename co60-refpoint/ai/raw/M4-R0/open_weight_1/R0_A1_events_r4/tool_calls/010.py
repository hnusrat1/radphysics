import numpy as np
import math
import random

mec2 = 0.511
r0 = 2.8179403262e-13
n_e_per_cm3 = 3.3456e23

source_z = -100.0
phantom_z_min = 0.0
phantom_z_max = 30.0
phantom_half = 15.0
scoring_r = 1.0
scoring_z0 = 5.0
scoring_dz = 0.1
field_half = 5.0

def mu_inc(E):
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

def mu_pp(E):
    if E <= 2*mec2:
        return 0.0
    e_arr = np.array([1.1732, 1.3325, 2.0, 3.0, 5.0, 10.0])
    mu_arr = np.array([0.006205, 0.006286, 0.0082, 0.0109, 0.0126, 0.0120])
    if E <= 1.1732:
        t = E - 2*mec2
        if t <= 0:            return 0.0        t0 = 1.1732 -1.022        return 0.006205 * (t/t0)**2    elif E >= 10.0:
        return 0.0120
    else:
        le = math.log(E)
        le_arr = np.log(e_arr)
        lmu_arr = np.log(mu_arr)
        return math.exp(n.interp(le, le_arr, lmu_arr))