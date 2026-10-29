import numpy as np
from scipy.integrate import quad
import math

# Physical constants
m_e = 0.511  # electron rest mass in MeV

# =========== Cross section calculations ===========

def klein_nishina_differential(epsilon, costheta):
    """
    Differential Klein-Nishina cross section per electron
    dσ/dΩ in cm^2/sr/electron
    epsilon = E/(m_e c^2) = E/0.511
    costheta = cos of scattering angle
    """
    mu = costheta
    r_e = 2.818e-13  # classical electron radius in cm
    
    # Energy ratio
    k = epsilon
    k_prime = k / (1 + k * (1 - mu))
    
    # Klein-Nishina formula
    ds_dOmega = (r_e**2 / 2) * (k_prime/k)**2 * (k/k_prime + k_prime/k - 1 + mu**2)
    return ds_dOmega  # cm^2/sr/electron

def compton_total_per_electron(epsilon):
    """Total Compton cross section per electron (cm^2)"""
    r_e = 2.818e-13
    k = epsilon
    
    def integrand(mu):
        k_prime = k / (1 + k * (1 - mu))
        return 0.5 * (k_prime/k)**2 * (k/k_prime + k_prime/k - 1 + mu**2)
    
    # Integrate over mu from -1 to 1, dΩ = 2π d(cosθ)
    integral, _ = quad(integrand, -1, 1)
    sigma = r_e**2 * 2 * np.pi * integral
    return sigma

def compton_scattered_energy(epsilon, costheta):
    """Energy of scattered photon in MeV"""
    return epsilon * m_e / (1 + epsilon * (1 - costheta))

def compton_electron_energy(epsilon, costheta):
    """Kinetic energy of recoil electron in MeV"""
    return epsilon * m_e - compton_scattered_energy(epsilon, costheta)

def compton_electron_angle(epsilon, costheta):
    """Cosine of electron angle relative to incident photon direction"""
    scattered_eps = epsilon / (1 + epsilon * (1 - costheta))
    # tan(theta_e) = cot(theta_photon/2) ... no, let me use correct formula
    # cosθ_e = (E_in - E_out * cosθ) / sqrt(E_in^2 + E_out^2 - 2*E_in*E_out*cosθ)
    E_in = epsilon * m_e
    E_out = scattered_eps * m_e
    # Actually: cosθ_e = (E_in - E_out) * sqrt((E_in - E_out + m_e)/(2*m_e*E_in*E_out))... hmm
    # Better: using momentum conservation:
    # p_e cosθ_e = p_in - p_out cosθ
    # p_e sinθ_e = p_out sinθ
    # tanθ_e = (p_out sinθ) / (p_in - p_out cosθ)
    # cosθ_e = (p_in - p_out cosθ) / sqrt(p_in^2 + p_out^2 - 2*p_in*p_out*cosθ)
    p_in = math.sqrt(E_in**2 + 2*E_in*m_e) if E_in > 0 else 0  # relativistic momentum
    p_out = math.sqrt(E_out**2 + 2*E_out*m_e) if E_out > 0 else 0
    
    if p_in == 0:
        return 0.0
    
    cos_e = (p_in - p_out * costheta) / math.sqrt(p_in**2 + p_out**2 - 2*p_in*p_out*costheta)
    # Clip to valid range
    cos_e = max(-1.0, min(1.0, cos_e))
    return cos_e

# Compute cross sections for 2.0 MeV
E0 = 2.0  # MeV
epsilon0 = E0 / m_e

# Water: 2 H + 1 O, density 1 g/cm^3
# Electrons per gram: Z/A * N_A
# Water molar mass: 18.015 g/mol
# Electrons per molecule: 10 (2*1 + 8)
# N_A = 6.022e23
N_A = 6.022e23
electrons_per_molecule = 10
molar_mass = 18.015  # g/mol
electrons_per_gram = N_A * electrons_per_molecule / molar_mass  # electrons/g
print(f"Electrons per gram of water: {electrons_per_gram:.6e}")

# Atoms per cm^3 (density 1 g/cm^3)
atoms_per_cm3 = N_A / molar_mass * 1.0  # molecules/cm^3
print(f"Molecules per cm^3: {atoms_per_cm3:.6e}")

# Total Compton cross section per electron at 2 MeV
sigma_comp_per_e = compton_total_per_electron(epsilon0)
print(f"Compton σ per electron (2 MeV): {sigma_comp_per_e:.6e} cm^2")

# Compton mass attenuation coefficient
mu_comp_rho = sigma_comp_per_e * electrons_per_gram
print(f"μ_comp/ρ (2 MeV): {mu_comp_rho:.6f} cm^2/g")

# Linear Compton attenuation coefficient
mu_comp = mu_comp_rho * 1.0  # density 1 g/cm^3
print(f"μ_comp (2 MeV): {mu_comp:.6f} cm^{-1}")

# Now add other processes. Let me estimate using known data.
# For water at 2 MeV:
# Photoelectric: very small
# Pair production: moderate
# Rayleigh: very small

# From NIST data (approximate):
# μ_photo/ρ at 2 MeV in water: ~0.00025 cm^2/g
mu_photo_rho = 0.00025
mu_photo = mu_photo_rho * 1.0

# μ_pair/ρ at 2 MeV in water: ~0.0078 cm^2/g
mu_pair_rho = 0.0078
mu_pair = mu_pair_rho * 1.0

# μ_rayleigh/ρ at 2 MeV in water: ~0.00006 cm^2/g
mu_rayleigh_rho = 0.00006
mu_rayleigh = mu_rayleigh_rho * 1.0

mu_total = mu_comp + mu_photo + mu_pair + mu_rayleigh
print(f"\nTotal attenuation coefficients:")
print(f"  Compton:    {mu_comp:.6f} cm^-1  ({mu_comp/mu_total*100:.2f}%)")
print(f"  Photoelec:  {mu_photo:.6f} cm^-1  ({mu_photo/mu_total*100:.2f}%)")
print(f"  Pair:       {mu_pair:.6f} cm^-1  ({mu_pair/mu_total*100:.2f}%)")
print(f"  Rayleigh:   {mu_rayleigh:.6f} cm^-1  ({mu_rayleigh/mu_total*100:.2f}%)")
print(f"  Total:      {mu_total:.6f} cm^-1")

# Attenuation factor for primary at 10 cm
print(f"\nPrimary attenuation at 10 cm: {np.exp(-mu_total*10):.4f}")

# Interaction probability in scoring volume (thickness 0.2 cm)
# For primary photons
p_interact_primary = 1 - np.exp(-mu_total * 0.2)
print(f"P(interact in scoring vol, primary): {p_interact_primary:.6f}")

# Relative interaction probabilities for 2.0 MeV
print(f"\nRelative interaction probabilities at 2.0 MeV:")
total = mu_total
print(f"  Compton:    {mu_comp/total:.4f}")
print(f"  Photoelec:  {mu_photo/total:.4f}")
print(f"  Pair:       {mu_pair/total:.4f}")
print(f"  Rayleigh:   {mu_rayleigh/total:.4f}")