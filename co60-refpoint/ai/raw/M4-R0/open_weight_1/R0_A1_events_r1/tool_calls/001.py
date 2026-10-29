import numpy as np
import math

# Constants
r_e = 2.8179403262e-13  # classical electron radius in cm
N_A = 6.02214076e23     # Avogadro's number
m_e_c2 = 0.5110034      # electron rest mass energy in MeV
H2O_M = 18.01528        # g/mol
H2O_rho = 1.0           # g/cm^3
H2O_Z_eff = 10          # electrons per molecule (2*1 + 8)

# electrons per cm^3 for water
n_e_per_cm3 = N_A * H2O_Z_eff * H2O_rho / H2O_M

# Klein-Nishina total cross section per electron
def sigma_KN(E_MeV):
    """Total Klein-Nishina cross section per electron (cm^2)"""
    alpha = E_MeV / m_e_c2
    a = alpha
    # Total KN cross section
    term = ( (1+a)/a**2 * (2*(1+a)/(1+2*a) - np.log(1+2*a)/a) + \
            np.log(1+2*a)/(2*a) - (1+3*a)/(1+2*a)**2
    return 2 * np.pi * r_e**2 * term

# Compute Compton attenuation coefficient
def mu_Compton(E_MeV):
    return n_e_per_cm3 * sigma_KN(E_MeV)

# Test
for E in [1.1732, 1.3325]:
    s = sigma_KN(E)
    mu_c = mu_Compton(E)
    print(f"E={E:.4f} MeV: sigma_KN={s:.4e} cm^2, mu_Compton={mu_c:.6f} cm^-1")
    print(f"  mu/rho_Compton={mu_c:.6f} cm^2/g")