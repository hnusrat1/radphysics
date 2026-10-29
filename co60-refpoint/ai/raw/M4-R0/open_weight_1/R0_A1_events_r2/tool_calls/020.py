import numpy as np

# Constants
me = 0.5109989461
r_e = 2.8179403227e-13
NA = 6.02214076e23
M_H2O = 18.015
N_e = 10
E1 = 1.1732
E2 = 1.3325

np.random.seed(12345)

# Klein-Nishina total cross section per electron
def sigma_KN(E):
    k = E / me
    term = (((1+k)/(k*k)) * (2*(1+k)/(1+2*k) - np.log(1+2*k)/k) +
            np.log(1+2*k)/(2*k) - (1+3*k)/(1+2*k)**2)
    return 2 * np.pi * r_e**2 * term

# Partial mu/rho (cm2/g)
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

# Sample Compton cos(theta)
def sample_compton_cos(E):
    alpha = E / me
    while True:
        mu = np.random.uniform(-1, 1)
        f = 1.0 / (1.0 + alpha * (1.0 - mu))
        dcs = f**2 * (f + 1.0/f - (1.0 - mu**2))
        if np.random.random() < dcs / 2.0:
            return mu

# Electron angle for Compton
def compton_electron_cos(E, cos_theta):
    sin_theta = np.sqrt(1 - cos_theta**2)
    tan_half = sin_theta / (1 + cos_theta)
    alpha = E / me
    cot_phi = (1 + alpha) * tan_half
    if cot_phi == 0:
        return 0.0
    tan_phi = 1.0 / cot_phi
    cos_phi = 1.0 / np.sqrt(1.0 + tan_phi**2)
    return cos_phi