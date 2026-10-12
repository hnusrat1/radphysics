import numpy as np
import math
import random

# Constants
rho = 1.0  # g/cm^3
N_A = 6.02214076e23
m_e_MeV = 0.51099895  # MeV
r_e = 2.8179403262e-13  # cm, classical electron radius

# Water: H2O, molar mass 18.015 g/mol, 10 electrons per molecule
A_water = 18.015
Z_water = 10  # electrons per molecule
n_e_cm3 = rho * N_A * Z_water / A_water  # electrons per cm^3

print(f"Electron density: {n_e_cm3:.4e} cm^-3")

# Klein-Nishina cross section per electron (cm^2)
def sigma_kn(E_MeV):
    alpha = E_MeV / m_e_MeV
    # Klein-Nishina formula
    term1 = (1 + alpha) / alpha**2 * (2*(1+alpha)/(1+2*alpha) - math.log(1+2*alpha)/alpha)
    term2 = math.log(1+2*alpha) / (2*alpha)
    term3 = (1 + 3*alpha) / (1+2*alpha)**2
    sigma = 2*math.pi * r_e**2 * (term1 + term2 - term3)
    return sigma

# Test at 2 MeV
print(f"Sigma KN at 2 MeV: {sigma_kn(2.0):.6e} cm^2")
print(f"Mu Compton at 2 MeV: {n_e_cm3 * sigma_kn(2.0):.6f} cm^-1")