import numpy as np

# Constants
me = 0.5109989461
r_e = 2.8179403227e-13  # cm
NA = 6.02214076e23
M_H2O = 18.015
N_e = 10
E1 = 1.1732
E2 = 1.3325

# Cross sections using correct Klein-Nishina
def sigma_KN(E):
    k = E / me
    term = ((1+k)/k**2)*(2*(1+k)/(1+2*k) - np.log(1+2*k)/k) + np.log(1+2*k)/(2*k) - (1+3*k)/(1+2*k)**2
    return 2*np.pi*r_e**2*term

# Test
print("sigma_KN at 1 MeV:", sigma_KN(1.0))

# Partial mu/rho for water
def partial_mu(E):
    # Compton (free electron)
    mu_comp = N_e * sigma_KN(E) * NA / M_H2O
    # Rayleigh: approximate from NIST at 1 MeV
    mu_rayl = 0.0014 * (1.0/max(E,0.01))**2
    # Photoelectric
    mu_pe = 0.0016 * (1.0/max(E,0.01))**3.5
    # Pair
    if E > 1.022:
        mu_pair = 0.00087 * (E-1.022)/(1.25-1.022)
    else:
        mu_pair = 0.0
    mu_tot = mu_comp + mu_rayl + mu_pe + mu_pair
    return mu_tot, mu_comp, mu_rayl, mu_pe, mu_pair

# Test at primary energies
for E in [E1, E2]:
    mt, mc, mr, mp, mpr = partial_mu(E)
    print(f"E={E}: tot={mt:.4f}, comp={mc:.4f}, rayl={mr:.4f}, pe={mp:.4f}, pair={mpr:.4f}"")

print("Done")