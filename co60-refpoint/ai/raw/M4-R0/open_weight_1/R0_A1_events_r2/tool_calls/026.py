import numpy as np

me = 0.5109989461
r_e = 2.8179403227e-13
NA = 6.02214076e23
M_H2O = 18.015
N_e = 10

def sigma_KN(E):
    k = E / me
    term1 = ((1+k)/(k*k)) * (2*(1+k)/(1+2*k) - np.log(1+2*k)/k)
    term2 = np.log(1+2*k)/(2*k)
    term3 = (1+3*k)/(1+2*k)**2
    return 2 * np.pi * r_e**2 * (term1 + term2 - term3)

def partial_mu(E):
    mu_comp = N_e * sigma_KN(E) * NA / M_H2O
    mu_rayl = 0.0014 * (1.0 / max(E, 0.01))**2
    mu_pe = 0.0016 * (1.0 / max(E, 0.01))**3.5
    if E > 1.022:
        mu_pair = 0.00087 * (E - 1.022) / (1.25 - 1.022)
    else:
        mu_pair = 0.0
    mu_tot = mu_comp + mu_rayl + mu_pe + mu_pair
    return mu_tot, mu_comp, mu_rayl, mu_pe, mu_pair

for E in [0.1, 0.2, 0.3, 0.5, 0.8]:
    mt, mc, mr, mp, mpr = partial_mu(E)
    print(f"E={E:.2f}: mu_tot={mt:.4f} cm2/g = {mt:.4f} cm-1")