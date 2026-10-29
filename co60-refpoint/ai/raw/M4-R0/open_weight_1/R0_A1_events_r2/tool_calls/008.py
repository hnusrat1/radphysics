import numpy as np
import time
from scipy.constants import physical_constants

# Constants
r_e = physical_constants['classical electron radius'][0] * 100  # cm
mec2 = 0.5109989461  # MeV
NA = 6.02214076e23  # mol^-1
M_H2O = 18.015  # g/mol
rho_water = 1.0  # g/cm3

# Number of electrons per water molecule
N_e = 10

# Klein-Nishina total cross section per electron (cm^2)
def sigma_KN(E):
    k = E / mec2
    term = ((1+k)/k**2 * (2*(1+k)/(1+2*k) - np.log(1+2*k)/k) 
            + np.log(1+2*k)/(2*k) 
            - (1+3*k)/(1+2*k)**2)
    return 2*np.pi * r_e**2 * term

# Klein-Nishina differential cross section normalized (dσ/dΩ in cm2/sr per electron)
# Returns the normalized differential cross section value for given incident energy and cos(theta)
def dsigma_KN_dOmega(E, mu):
    k = E / mec2
    f = 1.0 / (1.0 + k*(1.0 - mu))
    dcs = 0.5 * r_e**2 * f**2 * (f + 1.0/f - (1.0 - mu**2))
    return dcs

# Thomson cross section for Rayleigh (dσ/dΩ = r_e^2/2 * (1+mu^2) * |F(q,Z)|^2)
# For simplicity, we use the free atom form factor approximation
# We'll use the exponential model for form factor: F(q) = Z * exp(-a*(q/q0)^2)
# But for simplicity, we sample Rayleigh scattering with a very forward peaked distribution.

# For Rayleigh, we use the differential cross section:
# dσ/dΩ = r_e^2/2 * (1+mu^2) * [F(x,Z)]^2
# where x = sin(θ/2) / λ (in Å^-1) = (E/hc) * sin(θ/2) / (2π) ?
# Actually, the momentum transfer variable x = sin(θ/2) / λ, with λ = hc/E.
# For water, we can approximate the molecular form factor.

# For simplicity, I'll implement Rayleigh using the approximation that 
# the cross section is very small and the scattering is highly forward.
# I'll sample cosθ close to 1.

# Photoelectric: simple forward distribution
# Pair production: forward distribution

# Let's first compute the total cross sections per molecule using free-electron Compton
# and then add approximate Rayleigh, PE, pair.

def sigma_total_approx(E):
    """Approximate total cross section per water molecule (cm^2)"""
    # Compton (free electron)
    sig_comp = N_e * sigma_KN(E)
    
    # Rayleigh: approximate using the formula from NIST-like data
    # For water at 1 MeV, μ_rayleigh/ρ ≈ 0.0014 cm2/g
    # σ_Rayleigh = μ_rayleigh/ρ * M_H2O / NA
    # We'll approximate as proportional to Z^2 * (mec2/E)^2
    # At 1 MeV, target σ_Rayleigh = 0.0014 * 18.015 / 6.022e23 = 4.19e-26 cm2
    # At other energies, scale as (1 MeV/E)^2
    E_ref = 1.0  # MeV
    sig_rayl_target = 0.0014 * M_H2O / NA  # at 1 MeV
    sig_rayl = sig_rayl_target * (E_ref / max(E, 0.01))**2
    
    # Photoelectric: approximate using σ_PE ∝ Z^4.5 / E^3.5
    # At 1 MeV, target μ_PE/ρ = 0.0016 cm2/g
    sig_pe_target = 0.0016 * M_H2O / NA  # at 1 MeV
    sig_pe = sig_pe_target * (E_ref / max(E, 0.01))**3.5
    
    # Pair production: only if E > 1.022 MeV
    if E > 1.022:
        # Approximate σ_pair ∝ Z^2 * (E - 1.022)
        # At 1.25 MeV, target μ_pair/ρ = 0.00087 cm2/g
        # We'll calibrate at 1.25 MeV
        sig_pair_target = 0.00087 * M_H2O / NA  # at 1.25 MeV
        sig_pair = sig_pair_target * (E - 1.022) / (1.25 - 1.022)
    else:
        sig_pair = 0.0
    
    total = sig_comp + sig_rayl + sig_pe + sig_pair
    return total, sig_comp, sig_rayl, sig_pe, sig_pair

# Test at 1.1732 MeV
E_test = 1.1732
tot, comp, rayl, pe, pair = sigma_total_approx(E_test)
print(f"E={E_test} MeV")
print(f"σ_total = {tot:.4e} cm2")
print(f"σ_Compton = {comp:.4e}")
print(f"σ_Rayleigh = {rayl:.4e}")
print(f"σ_PE = {pe:.4e}")
print(f"σ_Pair = {pair:.4e}")
print(f"μ/ρ_total = {tot * NA / M_H2O:.4f} cm2/g")
print(f"Fractions: Compton={comp/tot*100:.2f}%, Rayleigh={rayl/tot*100:.2f}%, PE={pe/tot*100:.2f}%, Pair={pair/tot*100:.2f}%")