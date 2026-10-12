import numpy as np
from scipy import integrate

def compute_compton_energy_transfer_v2(E_MeV):
    """Compute mean energy transfer fraction from Klein-Nishina cross section"""
    mec2 = 0.511  # MeV
    alpha = E_MeV / mec2
    
    # The differential cross section dσ/d(cosθ) ∝ f(cosθ)
    # Energy transfer per interaction: E_transfer = E_γ - E'_γ
    # where E'_γ = E_γ / (1 + α(1-cosθ))
    
    n_points = 10000
    cos_theta = np.linspace(-1, 1, n_points)
    
    # Klein-Nishina function proportional to dσ/dΩ
    def KN(cos_t):
        # Photon energy after scattering
        Ep = 1.0 / (1.0 + alpha * (1.0 - cos_t))  # In units of original E
        r = Ep
        # Differential cross section factor
        return 0.5 * ((1.0 + cos_t**2) - alpha**2 * (1.0 - cos_t)**2 * (1.0 + r**2) / (1.0 + alpha*(1.0-cos_t))**2)
    
    # Total cross section (proportional)
    sigma_vals = np.array([KN(ct) for ct in cos_theta])
    sigma_total = integrate.trapezoid(sigma_vals, cos_theta)
    
    # Mean energy transfer (normalized by E)
    def energy_trans_frac(cos_t):
        Ep = 1.0 / (1.0 + alpha * (1.0 - cos_t))  # Normalized scattered photon energy
        transfer = 1.0 - Ep  # Fraction of energy transferred
        return transfer * KN(cos_t)
    
    t_vals = np.array([energy_trans_frac(ct) for ct in cos_theta])
    total_transfer = integrate.trapezoid(t_vals, cos_theta)
    
    if sigma_total > 0:
        return total_transfer / sigma_total
    return 0.4

# Test at different energies
for E in [0.5, 1.0, 1.253, 1.5, 2.0]:
    frac = compute_compton_energy_transfer_v2(E)
    print(f"E = {E:.2f} MeV: <E_tr/E> = {frac:.4f}")