import numpy as np
from scipy import constants

m_e = 0.510998950  # MeV/c^2
r_e = 2.8179403262e-13  # cm
N_A = 6.02214076e23

def klein_nishina_sigma(E_MeV):
    """Return Klein-Nishina cross section per electron (cm^2)"""
    alpha = E_MeV / m_e
    if alpha <= 0:
        return 0.0
    term1 = (1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha)
    term2 = np.log(1+2*alpha) / (2*alpha)
    term3 = (1+3*alpha) / (1+2*alpha)**2
    sigma = 2*np.pi * r_e**2 * (term1 + term2 - term3)
    return sigma

# For water
Z_H2O = 10
A_H2O = 18.01528  # g/mol
N_e_per_g = N_A * Z_H2O / A_H2O

def mu_compton(E_MeV):
    sigma = klein_nishina_sigma(E_MeV)
    return sigma * N_e_per_g

E = 2.0
print(f"Compton mu/rho at {E} MeV: {mu_compton(E):.6f} cm^2/g")

# Also compute total cross section using known values
# Let's also compute photoelectric, pair, Rayleigh using more accurate formulas or tables

# For photoelectric, let's use the formula from ICRU or Schenter & Hubbell
# Actually, let's use a simple power law fit to known NIST values

# NIST values for water (I'll look up from memory/documents)
# Actually I've seen these values before:
# Water: μ/ρ (cm^2/g)
# 2 MeV: total 0.04920, Compton 0.04906, PE 0.000016, pair 0.000127, Rayleigh ~0
# Let's check consistency

# My Klein-Nishina gave 0.048925 at 2 MeV, close to 0.04906
# So total ~ 0.0492

# Let's define functions that return mu/rho for each process
def get_mu_total(E):
    return mu_compton(E) + get_mu_pe(E) + get_mu_pair(E) + get_mu_rayleigh(E)

def get_mu_pe(E):
    # Approximate using known data
    # Photoelectric for water: σ_pe ~ Z^4.5 / E^3.5
    # Fit to NIST values:
    # At 0.1 MeV: mu_pe ~ 0.0017
    # At 0.2 MeV: mu_pe ~ 0.00027
    # At 0.5 MeV: mu_pe ~ 0.000025
    # At 1 MeV: ~ 0.000007
    # At 2 MeV: ~ 0.000002
    # Use interpolation
    if E < 0.01:
        return 4.89
    elif E < 0.1:
        # power law: 4.89 * (E/0.01)^(-3.5)
        return 4.89 * (E/0.01)**(-3.5)
    elif E < 1.0:
        # 0.0017 at 0.1, 7e-6 at 1.0
        return 0.0017 * (E/0.1)**(-3.2)
    else:
        # 7e-6 at 1.0, 2e-6 at 2.0
        return 7e-6 * (E/1.0)**(-3.5)

def get_mu_pair(E):
    # Pair production
    if E <= 2*m_e:
        return 0.0
    # Above threshold, rises gradually
    # At 2 MeV: ~ 0.00013
    # At 5 MeV: ~ 0.0015
    # At 10 MeV: ~ 0.0048
    # At 20 MeV: ~ 0.0077
    # Use interpolation
    if E < 2.0:
        return 0.0
    elif E < 5.0:
        # linear approx from 0.00013 at 2 to 0.0015 at 5
        return 0.00013 + (0.0015-0.00013)*(E-2)/3
    elif E < 10.0:
        return 0.0015 + (0.0048-0.0015)*(E-5)/5
    elif E < 20.0:
        return 0.0048 + (0.0077-0.0048)*(E-10)/10
    else:
        return 0.0077

def get_mu_rayleigh(E):
    # Rayleigh scattering
    # Negligible above ~0.5 MeV
    if E < 0.01:
        return 0.16
    elif E < 0.1:
        return 0.16 * (E/0.01)**(-3.5)
    elif E < 0.5:
        return 0.0002 * (E/0.1)**(-4)
    else:
        return 0.0

print(f"At 2 MeV:")
print(f"Compton: {mu_compton(2.0):.6f}")
print(f"PE: {get_mu_pe(2.0):.6e}")
print(f"Pair: {get_mu_pair(2.0):.6e}")
print(f"Rayleigh: {get_mu_rayleigh(2.0):.6e}")
print(f"Total: {mu_compton(2.0)+get_mu_pe(2.0)+get_mu_pair(2.0)+get_mu_rayleigh(2.0):.6f}")