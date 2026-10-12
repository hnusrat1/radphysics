import numpy as np
from numpy.random import default_rng

rng = default_rng(42)

# Constants
mec2 = 0.5110034
re = 2.8179403262e-13
nel_cm3 = 6.02214076e23 * 10 / 18.01528

def sigma_KN(E):
    a = E / mec2
    term = ((1+a)/a**2 * (2*(1+a)/(1+2*a) - np.log(1+2*a)/a) +
            np.log(1+2*a)/(2*a) - (1+3*a)/(1+2*a)**2)
    return 2 * np.pi * re**2 * term

def mu_c(E): return nel_cm3 * sigma_KN(E)
def mu_p(E): return 0.00057 * (1.0/E)**3.5
def mu_pp(E):
    if E <= 1.022:
        return 0.0
    return 0.00035 * (E-1.022)/(1.3325-1.022)
def mu_r(E): return 0.00033 * (1.0/E)**2
def mu_t(E): return mu_c(E)+mu_p(E)+mu_pp(E)+mu_r(E)

print("Cross sections at 1.1732:", [mu_c(1.1732), mu_p(1.1732), mu_pp(1.1732), mu_r(1.1732), mu_t(1.1732)])
print("Cross sections at 1.3325:", [mu_c(1.3325), mu_p(1.3325), mu_pp(1.3325), mu_r(1.3325), mu_t(1.3325)])