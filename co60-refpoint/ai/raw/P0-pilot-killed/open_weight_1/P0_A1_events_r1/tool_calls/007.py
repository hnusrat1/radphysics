import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.integrate import quad

# Constants
rho = 1.0  # g/cm3
me = 0.511  # MeV
r_e = 2.817940e-13  # cm
sigma_T = 8.0/3.0 * math.pi * r_e**2  # Thomson cross section

# Cross sections functions (already defined)
# We'll also need a function for total mu
def mu_total(E):
    return (mu_coh_func(E) + mu_inc_func(E) + mu_ph_func(E) + mu_pair_func(E))

# Test
print("mu_total at 2 MeV:", mu_total(2.0))
print("mu_total at 0.5 MeV:", mu_total(0.5))