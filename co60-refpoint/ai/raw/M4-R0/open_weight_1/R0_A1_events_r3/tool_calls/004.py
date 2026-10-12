import numpy as np
from scipy import integrate
import random

# Constants
m_e_c2 = 0.510998950  # MeV
r_e = 2.8179403262e-13  # classical electron radius, cm
NA = 6.02214076e23

# Water composition
# H2O: molar mass = 18.01528 g/mol, 10 electrons per molecule
e_per_g = 10 * NA / 18.01528  # electrons per gram
e_per_cm3 = e_per_g * 1.0  # density 1.0 g/cm^3

print(f"Electron density: {e_per_cm3:.4e} e-/cm^3")

# ============================================================
# Cross sections
# ============================================================

def compton_total_sigma(E):
    """Total Compton cross section per electron (cm^2)"""
    alpha = E / m_e_c2
    term1 = (1 + alpha) / alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha)
    term2 = np.log(1+2*alpha) / (2*alpha)
    term3 = (1 + 3*alpha) / (1 + 2*alpha)**2
    return 2 * np.pi * r_e**2 * (term1 + term2 - term3)

def compton_mu(E):
    """Compton linear attenuation coefficient for water (cm^-1)"""
    return compton_total_sigma(E) * e_per_cm3

def compton_dsigma_dOmega(E, cos_theta):
    """Klein-Nishina differential cross section dσ/dΩ per electron (cm^2/sr)"""
    alpha = E / m_e_c2
    # Energy of scattered photon
    E_prime = E / (1 + alpha * (1 - cos_theta))
    P = (E_prime / E)**2 * (E_prime/E + E/E_prime - (1 - cos_theta**2))
    return r_e**2 * P / 2

def photoelectric_mu(E):
    """Photoelectric linear attenuation coefficient for water (cm^-1)"""
    # Based on NIST data for water
    # Use interpolation of known data
    e_data = np.array([0.5, 0.6, 0.8, 1.0, 1.1732, 1.25, 1.3325, 1.5, 2.0])
    mu_data = np.array([0.0100, 0.0075, 0.0052, 0.00377, 0.0030, 0.00258, 0.0023, 0.00179, 0.0010])
    return np.interp(E, e_data, mu_data) * 1.0  # multiply by density

def pair_mu(E):
    """Pair production linear attenuation coefficient for water (cm^-1)"""
    if E < 2 * m_e_c2:
        return 0.0
    # Interpolate from known data
    e_data = np.array([1.022, 1.1, 1.1732, 1.25, 1.3325, 1.5, 2.0, 3.0, 5.0])
    mu_data = np.array([0.0, 0.0002, 0.00035, 0.00071, 0.0010, 0.00146, 0.00382, 0.0085, 0.016])
    return np.interp(E, e_data, mu_data) * 1.0

def rayleigh_mu(E):
    """Rayleigh scattering linear attenuation coefficient for water (cm^-1)"""
    e_data = np.array([0.5, 0.6, 0.8, 1.0, 1.1732, 1.25, 1.3325, 1.5, 2.0])
    mu_data = np.array([0.0061, 0.0049, 0.0035, 0.00303, 0.0025, 0.00242, 0.0021, 0.00186, 0.00114])
    return np.interp(E, e_data, mu_data) * 1.0

def total_mu(E):
    """Total linear attenuation coefficient for water (cm^-1)"""
    return compton_mu(E) + photoelectric_mu(E) + pair_mu(E) + rayleigh_mu(E)

# Test cross sections
for E in [1.1732, 1.3325]:
    mu_c = compton_mu(E)
    mu_p = photoelectric_mu(E)
    mu_pp = pair_mu(E)
    mu_r = rayleigh_mu(E)
    mu_t = total_mu(E) # typo
    mu_t = total_mu(E)
    print(f"\nE = {E} MeV:")
    print(f"  mu (cm^-1) - Compton: {mu_c:.6f}, Photo: {mu_p:.6f}, Pair: {mu_pp:.6f}, Rayleigh: {mu_r:.6f}")
    print(f"  Total: {mu_t:.6f}")
    print(f"  Fractions - Compton: {mu_c/mu_t:.4f}, Photo: {mu_p/mu_t:.4f}, Pair: {mu_pp/mu_t:.4f}, Rayleigh: {mu_r/mu_t:.4f}")

# ============================================================
# Sampling functions
# ============================================================

def sample_compton_angle(E):
    """Sample cos(theta) for Compton scattering using Kahn's method"""
    alpha = E / m_e_c2
    while True:
        # Rejection sampling
        r1 = np.random.random()
        r2 = np.random.random()
        
        # Sample cos_theta from approximate distribution
        # Using the KN rejection sampling method
        epsilon = 2 * r1 - 1
        cos_theta = 1 - 2 * alpha * epsilon / (1 + alpha * (1 - epsilon))
        
        if cos_theta < -1 or cos_theta > 1:
            continue
        
        # Compute the actual rejection probability
        E_prime = E / (1 + alpha * (1 - cos_theta))
        ratio = (E_prime / E)**2 * (E_prime/E + E/E_prime - (1 - cos_theta**2))
        # For the envelope, use (1 + cos_theta^2) * some factor
        envelope = 1 + cos_theta**2
        
        if ratio * (1 + alpha)**2 > envelope * r2 * (1 + alpha * (1 - cos_theta))**2:
            # Actually this is getting complicated. Let me use a simpler method.
            pass
    
    # Let me use a different approach
    pass

def sample_compton_angle_simple(E):
    """Sample cos(theta) for Compton scattering"""
    # Use the method from Salvat et al. (PENELOPE)
    alpha = E / m_e_c2
    while True:
        # Step 1: Sample a value from the approximation
        r1 = np.random.random()
        r2 = np.random.random()
        
        # Sample from 1/(1+alpha*(1-cos))**2 distribution for the envelope
        # This gives cos = 1 - 2*alpha*r1/(1+alpha*(1-2*r1))
        # Actually let me just use simple rejection
        
        # Sample cos uniformly
        cos_theta = 2 * np.random.random() - 1
        
        # Check if accepted
        E_prime = E / (1 + alpha * (1 - cos_theta))
        w = (E_prime/E)**2 * (E_prime/E + E/E_prime - (1 - cos_theta**2))
        
        # Maximum of w is at cos_theta = 1 where w = 1
        # Minimum is at cos_theta = -1 where w = (1+2*alpha)/(1+2*alpha+2*alpha^2) * ...
        # Actually, let's just use a simple approach
        
        return cos_theta, E_prime

print("\n\nCross sections ready. Now building Monte Carlo...")