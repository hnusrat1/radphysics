import numpy as np
from scipy import interpolate

# Constants
m_e = 0.510998950  # MeV
r_e = 2.8179403262e-13  # cm
N_A = 6.02214076e23

# Water composition
# Density = 1 g/cm^3
# H2O: Z=10, A=18.01528 g/mol
# Number of electrons per g
N_e_per_g = N_A * 10 / 18.01528

# ---- Cross sections ----
def klein_nishina_sigma(E_MeV):
    """Klein-Nishina cross section per electron (cm^2)"""
    alpha = E_MeV / m_e
    if alpha <= 0:
        return 0.0
    term1 = (1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha)
    term2 = np.log(1+2*alpha) / (2*alpha)
    term3 = (1+3*alpha) / (1+2*alpha)**2
    sigma = 2*np.pi * r_e**2 * (term1 + term2 - term3)
    return sigma

def mu_compton(E):
    """Compton mass attenuation coefficient (cm^2/g)"""
    return klein_nishina_sigma(E) * N_e_per_g

# Photoelectric: use fit to NIST data
# We'll create interpolation for efficiency
pe_energies = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
pe_murho = np.array([4.89, 1.23, 0.359, 0.0559, 0.0150, 0.0050, 0.0019, 0.00038, 0.000163, 0.000044, 0.000020, 0.000008, 0.000004, 0.0000025, 0.0000018, 0.0000010, 0.0000007, 0.0000003, 0.0000002, 0.0000001, 0.00000003, 0.00000001])
log_pe_energies = np.log(pe_energies[1:])
log_pe_murho = np.log(pe_murho[1:])
f_pe_log = interpolate.interp1d(log_pe_energies, log_pe_murho, kind='linear', bounds_error=False, fill_value='extrapolate')
def mu_pe(E):
    if E <= 0:
        return 0.0
    if E < 0.01:
        # extrapolate at low end
        return pe_murho[0] * (E/0.01)**(-3.5)
    if E > 10.0:
        return 1e-10
    return np.exp(f_pe_log(np.log(E)))

# Pair production: interpolation
pair_energies = np.array([1.022, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0])
pair_murho = np.array([0.0, 0.00004, 0.00013, 0.00046, 0.0015, 0.0048, 0.0077])
f_pair = interpolate.interp1d(pair_energies, pair_murho, kind='linear', bounds_error=False, fill_value=0.0)
def mu_pair(E):
    if E <= 2*m_e:
        return 0.0
    if E > 20.0:
        return 0.0077 + 0.00002 * (E-20)
    return float(f_pair(E))

# Rayleigh: negligible above 0.5 MeV, but for low energies
ray_energies = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
ray_murho = np.array([0.16, 0.026, 0.0018, 0.0002, 0.00002, 0.000005, 0.000001])
# Actually let's just approximate
def mu_rayleigh(E):
    if E < 0.01:
        return 0.16
    elif E < 0.1:
        return 0.16 * (E/0.01)**(-3.5)
    elif E < 0.5:
        return 0.0002 * (E/0.1)**(-4)
    else:
        return 0.0

def mu_total(E):
    return mu_compton(E) + mu_pe(E) + mu_pair(E) + mu_rayleigh(E)

# Test
print("Cross sections at 2 MeV:")
print(f"Compton: {mu_compton(2.0):.6f}")
print(f"PE: {mu_pe(2.0):.6e}")
print(f"Pair: {mu_pair(2.0):.6e}")
print(f"Rayleigh: {mu_rayleigh(2.0):.6e}")
print(f"Total: {mu_total(2.0):.6f}")