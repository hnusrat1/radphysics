import numpy as np
from scipy.constants import physical_constants

# constants
r_e = physical_constants['classical electron radius'][0]  # cm
mec2 = physical_constants['electron mass energy equivalent in MeV'][0]  # MeV
NA = physical_constants['Avogadro constant'][0]  # 1/mol
# water
M_H2O = 18.015  # g/mol
# number of electrons per water molecule
Z_H = 1
Z_O = 8
N_electrons = 2*Z_H + Z_O  # = 10
# number of atoms per molecule
N_atoms = 2 + 1  # H, O

def klein_nishina_total(energy_MeV):
    """Total Klein-Nishina cross section per electron (cm^2)"""
    k = energy_MeV / mec2  # k = h nu / mec2
    # formula from Attix
    sigma = 2*np.pi*r_e**2 * ((1+k)/k**2 * (2*(1+k)/(1+2*k) - np.log(1+2*k)/k) + np.log(1+2*k)/(2*k) - (1+3*k)/(1+2*k)**2)
    return sigma

# compute for 1.1732 and 1.3325 MeV
for E in [1.1732, 1.3325]:
    sigma_per_e = klein_nishina_total(E)
    # per molecule
    sigma_mol = sigma_per_e * N_electrons
    # mass attenuation coefficient (cm2/g)
    mu_rho = sigma_mol * NA / M_H2O
    print(f"E={E} MeV: sigma_per_e={sigma_per_e:.4e} cm2, mu_rho_Compton={mu_rho:.6e} cm2/g")