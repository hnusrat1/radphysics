import numpy as np
import math
from numpy.random import default_rng

rng = default_rng(42)  # reproducible

# ============================================================
# Constants
# ============================================================
m_e_c2 = 0.5110034   # MeV
r_e = 2.8179403262e-13  # cm
N_A = 6.02214076e23
H2O_M = 18.01528      # g/mol
n_e_per_cm3 = N_A * 10 * 1.0 / H2O_M  # elec/cm^3 in water (rho=1)

SSD = 100.0           # cm
field_half = 5.0       # field is 10x10, half width 5cm
phantom_depth = 30.0   # cm
phantom_half = 15.0    # half lateral size

# Scoring volume
sc_z0 = 5.0           # centre depth (cm)
sc_rad = 1.0           # radius (cm)
sc_half = 0.1          # half-thickness (cm), so total thickness 0.2cm
sc_z_min = sc_z0 - sc_half
sc_z_max = sc_z0 + sc_half

# Source position
src = np.array([0.0, 0.0, -SSD])

# ============================================================
# Cross sections (cm^-1) for water at the two energies
# ============================================================
# We'll compute Compton from Klein-Nishina, others from approximations

def sigma_KN(E_MeV):
    """Klein-Nishina total cross section per electron (cm^2)"""
    a = E_MeV / m_e_c2
    term = ((1+a)/a**2 * (2*(1+a)/(1+2*a) - np.log(1+2*a)/a) +
            np.log(1+2*a)/(2*a) - (1+3*a)/(1+2*a)**2)
    return 2 * np.pi * r_e**2 * term

def mu_comp(E):
    return n_e_per_cm3 * sigma_KN(E)

def mu_photo(E):
    # Approx: at 1 MeV, mu_photo ≈ 0.00057 cm^-1, scaling ~ E^-3.5
    return 0.00057 * (1.0 / E)**3.5

def mu_pair(E):
    # Only above 1.022 MeV
    if E <= 1.022:
        return 0.0
    # Approx: at 1.3325 MeV, mu_pair ≈ 0.00035 cm^-1
    # Scaling ~ (E-1.022)/E * something
    # Use simple linear above threshold
    return 0.00035 * (E - 1.022) / (1.3325 - 1.022)

def mu_rayleigh(E):
    # Approx: at 1 MeV, mu_Rayleigh ≈ 0.00033, scaling ~ E^-2
    return 0.00033 * (1.0 / E)**2

# Test
for E in [1.1732, 1.3325]:
    mc = mu_comp(E)
    mp = mu_photo(E)
    mpr = mu_pair(E)
    mr = mu_rayleigh(E)
    mt = mc + mp + mpr + mr
    print(f"E={E:.4f} MeV:"
          f" comp={mc:.6f} photo={mp:.6f} pair={mpr:.6f} rayleigh={mr:.6f}"
          f" total={mt:.6f} cm^-1")