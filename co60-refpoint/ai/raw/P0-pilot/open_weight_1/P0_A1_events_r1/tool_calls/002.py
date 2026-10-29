import numpy as np
import math
from scipy.integrate import quad
from numpy import random

# Constants
m_e = 0.511  # MeV
N_A = 6.022e23
molar_mass_H2O = 18.015  # g/mol
electrons_per_H2O = 10
atoms_per_cm3 = N_A / molar_mass_H2O * 1.0  # molecules/cm^3
electrons_per_cm3 = atoms_per_cm3 * electrons_per_H2O

# Source parameters
SSD = 100.0  # cm
field_x = 10.0  # cm at phantom surface
field_y = 10.0  # cm
E0 = 2.0  # MeV

# Phantom dimensions
phantom_x = 30.0  # cm
phantom_y = 30.0
phantom_z = 30.0  # depth direction

# Scoring volume
scoring_center_depth = 10.0  # cm
scoring_radius = 1.0  # cm
scoring_thickness = 0.2  # cm

#=============================================================================
# Cross section data - let me compute more carefully
#=============================================================================

def compton_cs_per_electron(E_MeV):
    """Total Compton cross section per electron (cm^2)"""
    epsilon = E_MeV / m_e
    r_e = 2.818e-13
    
    # Klein-Nishina integrated over all angles
    k = epsilon
    if k == 0:
        return 0
    
    # The total Klein-Nishina cross section:
    # σ_KN = 2πr_e^2 * [ (1+k)/k^2 * ( 2(1+k)/(1+2k) - ln(1+2k)/k ) + ln(1+2k)/(2k) - (1+3k)/(1+2k)^2 ]
    sigma = 2 * np.pi * r_e**2 * (
        (1 + k) / k**2 * (2 * (1 + k) / (1 + 2*k) - np.log(1 + 2*k) / k) +
        np.log(1 + 2*k) / (2*k) - (1 + 3*k) / (1 + 2*k)**2
    )
    return sigma

def compton_mu(E_MeV):
    """Linear Compton attenuation coefficient for water (cm^-1)"""
    sigma_per_e = compton_cs_per_electron(E_MeV)
    return sigma_per_e * electrons_per_cm3

def compton_mu_rho(E_MeV):
    """Mass Compton attenuation coefficient (cm^2/g)"""
    return compton_mu(E_MeV) / 1.0  # density = 1

def compton_scatter(E_in, costheta):
    """
    Compton scattering: return (E_out, cos_electron, t_e)
    E_in: incident photon energy in MeV
    costheta: cosine of photon scattering angle
    """
    epsilon = E_in / m_e
    E_out = E_in / (1 + epsilon * (1 - costheta))
    t_e = E_in - E_out
    
    # Electron angle
    if E_in > 0 and E_out > 0:
        p_in = math.sqrt(E_in**2 + 2*E_in*m_e)
        p_out = math.sqrt(E_out**2 + 2*E_out*m_e)
        cos_e = (p_in - p_out * costheta) / math.sqrt(p_in**2 + p_out**2 - 2*p_in*p_out*costheta)
        cos_e = max(-1.0, min(1.0, cos_e))
    else:
        cos_e = 0.0
    
    return E_out, cos_e, t_e

def sample_compton_angle(epsilon):
    """
    Sample cos(theta) from Klein-Nishina distribution using rejection sampling
    epsilon = E/(m_e c^2)
    """
    r_e = 2.818e-13
    k = epsilon
    
    # Calculate maximum of dσ/dΩ for scaling
    # The function is (k'/k)^2 * (k/k' + k'/k - 1 + mu^2)
    # where k' = k/(1+k(1-mu))
    # Maximum occurs near mu=1 for high energies
    
    def ds_dOmega(mu):
        kp = k / (1 + k * (1 - mu))
        if kp <= 0:
            return 0
        return (kp/k)**2 * (k/kp + kp/k - 1 + mu**2)
    
    # Find approximate maximum
    mus_test = np.linspace(-1, 1, 1000)
    vals = np.array([ds_dOmega(mu) for mu in mus_test])
    max_val = np.max(vals) * 1.1
    
    # Rejection sampling
    while True:
        mu = random.uniform(-1, 1)
        u = random.uniform(0, 1)
        if u * max_val <= ds_dOmega(mu):
            return mu

#=============================================================================
# Other processes
#=============================================================================

def rayleigh_cs(E_MeV):
    """Rayleigh scattering cross section for water (cm^-1) - approximate"""
    # Rayleigh scattering cross section is very small at high energies
    # At 2 MeV it's ~0.00006 cm^-1 for water
    # Approximate with 1/E^2 dependence
    if E_MeV < 0.01:
        return 0.0
    # Approximate: at 100 keV it's about 0.06 cm^-1
    # Scale as ~1/E^2 for low energies, use approximate form
    # Very rough approximation for water
    # Actually let me just use a simple parameterization
    # Rayleigh scattering cross section varies as ~ Z^2/E^2
    # For water at 2 MeV: ~0.00006 cm^-1
    # Let me use a simple formula that gives approximately correct values
    sigma = 2.4e-25 * (1/E_MeV**2) * 1.0  # very rough
    sigma_per_cm3 = sigma * atoms_per_cm3  # per cm^3
    return sigma_per_cm3 * 1e-10  # fudge to get ~0.00006 at 2 MeV... 
    # Actually this is too rough. Let me just use known values.

# Let me compute using known parameterizations
def rayleigh_coherent_mu(E_MeV):
    """Coherent (Rayleigh) scattering linear attenuation for water"""
    # From NIST data, approximate fit for water
    # σ_coherent ~ Z^2 * f(x) / E^2 where f(x) is the form factor
    # Very rough but let me use tabulated approximate values
    # For water at various energies (cm^-1):
    # 0.01 MeV: ~8.0
    # 0.1 MeV: ~0.06
    # 1.0 MeV: ~0.0003
    # 2.0 MeV: ~0.00006
    # 10 MeV: ~0.000001
    if E_MeV < 0.01:
        return 8.0 * (0.01/E_MeV)**2.5  # rough
    elif E_MeV < 0.1:
        # interpolate
        return 8.0 * (0.01/E_MeV)**2.2
    elif E_MeV < 1.0:
        return 0.06 * (0.1/E_MeV)**2.3
    else:
        return 0.0003 * (1.0/E_MeV)**2.5

def photoelectric_mu(E_MeV):
    """Photoelectric linear attenuation for water"""
    # For water, approximate fit
    # At 0.01 MeV: ~10 cm^-1
    # At 0.1 MeV: ~0.04 cm^-1
    # At 0.5 MeV: ~0.001 cm^-1
    # At 1.0 MeV: ~0.0004 cm^-1
    # At 2.0 MeV: ~0.00025 cm^-1
    if E_MeV < 0.01:
        return 10.0 * (0.01/E_MeV)**3.0
    elif E_MeV < 0.1:
        return 10.0 * (0.01/E_MeV)**2.8
    elif E_MeV < 1.0:
        return 0.04 * (0.1/E_MeV)**2.5
    else:
        return 0.0004 * (1.0/E_MeV)**2.8

def pair_production_mu(E_MeV):
    """Pair production linear attenuation for water"""
    # Threshold at 1.022 MeV
    if E_MeV < 1.022:
        return 0.0
    
    # For water, at 2 MeV: ~0.0078 cm^-1
    # At 5 MeV: ~0.015 cm^-1
    # At 10 MeV: ~0.020 cm^-1
    # Roughly linear above threshold
    if E_MeV <= 2.0:
        # linear from threshold
        return 0.0 if E_MeV < 1.022 else 0.0078 * (E_MeV - 1.022) / (2.0 - 1.022)
    else:
        return 0.0078 + 0.002 * (E_MeV - 2.0)

def total_mu(E_MeV):
    """Total linear attenuation coefficient for water"""
    return (compton_mu(E_MeV) + 
            photoelectric_mu(E_MeV) + 
            pair_production_mu(E_MeV) + 
            rayleigh_coherent_mu(E_MeV))

#=============================================================================
# Check values at 2 MeV
#=============================================================================
E = 2.0
print(f"At E = {E} MeV:")
print(f"  Compton μ:       {compton_mu(E):.6f} cm^-1")
print(f"  Photoelectric μ: {photoelectric_mu(E):.6f} cm^-1")
print(f"  Pair production μ: {pair_production_mu(E):.6f} cm^-1")
print(f"  Rayleigh μ:      {rayleigh_coherent_mu(E):.6f} cm^-1")
print(f"  Total μ:         {total_mu(E):.6f} cm^-1")

# Relative probabilities
mu_comp = compton_mu(E)
mu_photo = photoelectric_mu(E)
mu_pair = pair_production_mu(E)
mu_ray = rayleigh_coherent_mu(E)
mu_tot = total_mu(E)
print(f"\nRelative interaction probabilities at {E} MeV:")
print(f"  Compton:    {mu_comp/mu_tot:.4f}")
print(f"  Photoelec:  {mu_photo/mu_tot:.4f}")
print(f"  Pair:       {mu_pair/mu_tot:.4f}")
print(f"  Rayleigh:   {mu_ray/mu_tot:.4f}")

# Also check at lower energies for scattered photons
for E_test in [0.1, 0.3, 0.5, 1.0, 1.5]:
    mt = total_mu(E_test)
    print(f"\nE = {E_test} MeV:")
    print(f"  Compton:    {compton_mu(E_test)/mt:.4f}, μ={compton_mu(E_test):.6f}")
    print(f"  Photoelec:  {photoelectric_mu(E_test)/mt:.4f}, μ={photoelectric_mu(E_test):.6f}")
    print(f"  Pair:       {pair_production_mu(E_test)/mt:.4f}, μ={pair_production_mu(E_test):.6f}")
    print(f"  Rayleigh:   {rayleigh_coherent_mu(E_test)/mt:.4f}, μ={rayleigh_coherent_mu(E_test):.6f}")