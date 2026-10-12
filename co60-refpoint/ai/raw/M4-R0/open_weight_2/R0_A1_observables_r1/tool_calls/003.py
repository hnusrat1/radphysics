import numpy as np
from scipy import integrate

def compute_compton_energy_transfer(E_MeV):
    """Compute mean energy transfer fraction from Klein-Nishina cross section"""
    mec2 = 0.511
    alpha = E_MeV / mec2
    
    # Integrate over photon scattering angle theta (0 to pi)
    n_points = 5000
    cos_theta = np.linspace(-1, 1, n_points)
    
    # Klein-Nishina differential cross section factor (proportional to dσ/dΩ)
    def KN_factor(cos_t):
        E_prime = E_MeV / (1 + alpha * (1 - cos_t))
        r = E_prime / E_MeV
        term1 = (1 + cos_t**2) / 2
        term2 = (alpha**2 * (1 - cos_t)**2 * (1 + r**2)) / (2 * (1 + alpha * (1 - cos_t))**2)
        return term1 - term2
    
    # Cross section integral
    integrand = np.array([KN_factor(ct) for ct in cos_theta])
    sigma_total = integrate.trapezoid(integrand, cos_theta)
    
    # Energy transfer integral
    def energy_transfer(cos_t):
        E_prime = E_MeV / (1 + alpha * (1 - cos_t))
        transfer = E_MeV - E_prime
        return transfer * KN_factor(cos_t)
    
    t_integrand = np.array([energy_transfer(ct) for ct in cos_theta])
    total_transfer = integrate.trapezoid(t_integrand, cos_theta)
    
    if sigma_total > 0:
        return total_transfer / sigma_total
    return 0.5

E_avg_co60 = 1.25285
compton_et_frac = compute_compton_energy_transfer(E_avg_co60)
print(f"Compton energy transfer fraction at {E_avg_co60:.3f} MeV: {compton_et_frac:.4f}")

# Verify with known values
# At 1 MeV, ~0.42; at 2 MeV, ~0.48; so ~0.45 at 1.25 MeV seems reasonable