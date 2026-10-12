import numpy as np
from scipy.constants import physical_constants

r_e_m = physical_constants['classical electron radius'][0]  # meters
r_e = r_e_m * 100  # convert to cm
mec2 = 0.5109989461  # MeV
NA = 6.02214076e23  # 1/mol
M_H2O = 18.015  # g/mol
N_electrons = 10

def klein_nishina_total(energy_MeV):
    alpha = energy_MeV / mec2
    term = ((1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha) 
            + np.log(1+2*alpha)/(2*alpha) 
            - (1+3*alpha)/(1+2*alpha)**2)
    sigma = 2 * np.pi * r_e**2 * term
    return sigma

for E in [1.1732, 1.3325]:
    sigma_per_e = klein_nishina_total(E)
    sigma_mol = sigma_per_e * N_electrons
    mu_rho_Compton = sigma_mol * NA / M_H2O
    print(f"E={E} MeV: sigma_per_e={sigma_per_e:.4e} cm2, mu_rho_Compton={mu_rho_Compton:.6e} cm2/g")