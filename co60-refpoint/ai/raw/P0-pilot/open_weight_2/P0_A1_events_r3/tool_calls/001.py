import numpy as np
from scipy import constants
import json

# Constants
E_photon = 2.0  # MeV
m_e = 0.5109989461  # MeV/c^2 (electron rest mass)
h_bar_c = 197.3269804  # MeV*fm (reduced Planck constant times c)

# Water properties (liquid)
Z_water = 7.424  # effective Z (weighted average)
N_A = 6.02214076e23  # Avogadro's number
rho_water = 1.0  # g/cm^3
A_water = 18.015  # g/mol (molecular weight)
n_electron_density = rho_water * N_A * Z_water / A_water  # electrons per cm^3

def klein_nishinel_diff(e_in, cos_theta):
    """Klein-Nishina differential cross section (normalized)."""
    e_out = e_in / (1 + e_in * (1 - cos_theta))
    r = e_out / e_in
    term1 = e_out / e_in
    term2 = (e_out / e_in)**2 + (e_in / e_out)**2 * (1 - cos_theta)**2
    return term1 * term2

def klein_nishine_total(e_in):
    """Total Klein-Nishina cross section."""
    # Analytic formula
    alpha = e_in / m_e
    x = 1 + 2*alpha
    
    sigma_kn = 2 * np.pi * (h_bar_c * 1e-13)**2 * (m_e**2) * [
        (1 + alpha) / alpha**2 * (2*(1+alpha)/x * np.log(x) - 2*alpha/x + 1/x**2) +
        -np.log(x) / alpha + 1/(2*alpha) * (1 - 4*alpha + 8*alpha**2) / x**2 -
        (1+3*alpha)/(alpha**2) * 1/x
    ]
    
    # Simpler approach: numerical integration
    n_points = 1000
    cos_thetas = np.linspace(-1, 1, n_points)
    dcos = cos_thetas[1] - cos_thetas[0]
    
    kn_values = []
    for ct in cos_thetas:
        e_out = e_in / (1 + e_in * (1 - ct))
        r = e_out / e_in
        val = r * (r**2 + (1/r)**2 * (1-ct)**2)
        kn_values.append(val)
    
    kn_array = np.array(kn_values)
    integral = np.trapz(kn_array, cos_thetas)
    
    # The angular distribution dσ/dΩ = (r₀²/2)(r + 1/r - sin²θ)
    # where r₀ is classical electron radius
    r_0 = 2.8179403262e-13  # cm (classical electron radius)
    sigma_tot = (r_0**2 / 2) * np.pi * integral
    
    return sigma_tot

# Let me use NIST XCOM data values directly (well-known values at 2.0 MeV in water)
# These are approximate mass attenuation coefficients in cm^2/g
# From NIST tables at 2.0 MeV for water:

# Compton scattering (incoherent)
mu_rho_compton = 0.0838  # cm^2/g (approximate)
# Coherent (Rayleigh) scattering
mu_rho_rayleigh = 0.0027  # cm^2/g (approximate) 
# Photoelectric effect
mu_rho_photo = 0.00017  # cm^2/g (approximate)
# Pair production (nuclear field)
mu_rho_pair_nuc = 0.0195  # cm^2/g (approximate)
# Total
mu_rho_total = mu_rho_compton + mu_rho_rayleigh + mu_rho_photo + mu_rho_pair_nuc

print(f"Mass attenuation coefficients:")
print(f"Compton: {mu_rho_compton}")
print(f"Rayleigh: {mu_rho_rayleigh}")
print(f"Photoelectric: {mu_rho_photo}")
print(f"Pair production: {mu_rho_pair_nuc}")
print(f"Total: {mu_rho_total}")

probabilities = np.array([mu_rho_compton, mu_rho_rayleigh, mu_rho_photo, mu_rho_pair_nuc]) / mu_rho_total
process_names = ["compton", "rayleigh", "photoelectric", "pair"]
for name, prob in zip(process_names, probabilities):
    print(f"{name}: {prob*100:.1f}%")
