import numpy as np
import scipy.integrate as integrate
import math

# Constants
mec2 = 0.511  # MeV
r0 = 2.8179403262e-13  # cm
r0_sq = r0**2
NA = 6.02214076e23
M_H2O = 18.01528
electrons_per_gram_H2O = 10 * NA / M_H2O

# Compton differential cross section
def dsigma_dOmega(E, costheta):
    """dσ/dΩ per electron (cm^2/sr), Klein-Nishina"""
    alpha = E / mec2
    k = 1.0 / (1.0 + alpha * (1.0 - costheta))
    Ep = E * k
    return (r0_sq/2.0) * k**2 * (k + 1.0/k - (1.0 - costheta**2)), Ep

def sigma_C(E):
    """Total Compton cross section per electron (cm^2)"""
    f, _ = dsigma_dOmega(E, np.linspace(-1, 1, 1000))
    # integrate dΩ = 2π d(costheta) from -1 to 1
    return 2*np.pi * integrate.simps(f, np.linspace(-1, 1, 1000))

# Check
print(f"σ_C(2 MeV) via integration: {sigma_C(2.0):.6e} cm^2")
print("(Expected ~1.46e-25 from analytic formula)")

# Average energy transfer
def compton_energy_quantities(E):
    """Return (sigma_total, avg_E_scattered, avg_E_transferred)"""
    costheta = np.linspace(-1, 1, 100000)
    dσ, Ep = dsigma_dΩ(E, costheta)
    # dΩ weight: 2π
    weight = 2*np.pi * np.ones_like(costheta)
    sigma = integrate.simps(dσ * weight / 2, costheta)  # hmm, dΩ = 2π d(cosθ), so integrate dσ * 2π d(cosθ)
    # Actually, let me just integrate dσ over cosθ with weight 2π
    sigma = integrate.simps(2*np.pi * dσ, costheta)
    avg_Ep = integrate.simps(2*np.pi * dσ * Ep, costheta) / sigma
    avg_E_transfer = E - avg_Ep
    return sigma, avg_Ep, avg_E_transfer

sigma, avg_Ep, avg_ET = compton_energy_quantities(2.0)
print(f"\nCompton at 2 MeV:")
print(f"  σ per electron: {sigma:.6e} cm^2")
print(f"  Average scattered photon energy: {avg_Ep:.4f} MeV")
print(f"  Average energy transferred to electron: {avg_ET:.4f} MeV")
print(f"  Fraction energy transferred: {avg_ET/2.0:.4f}")

# Also compute the distribution of initial electron energies
# For the initial spectrum in energy bins
def compton_electron_spectrum(E_photon, e_edges):
    """Compute the spectrum of electron kinetic energies from Compton scattering"""
    # For a photon of energy E, scattering at angle θ gives electron kinetic energy:
    # T = E - E' = E * alpha*(1-cosθ) / (1 + alpha*(1-cosθ))
    # where alpha = E/mec^2
    # dσ/dT distribution
    
    costheta = np.linspace(-1, 1, 200000)
    dσ, Ep = dsigma_dΩ(E_photon, costheta)
    T = E_photon - Ep
    
    # histogram using the dσ as weight
    # dσ/dT dT = dσ/dΩ * dΩ/dT * dT
    # but easier: just histogram weighted by dσ*2π
    
    # Use np.histogram
    counts, _ = np.histogram(T, bins=e_edges, weights=2*np.pi*dσ)
    return counts  # weighted count in each bin, proportional to cross section contribution

# Test
e_edges_test = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.4])
spectrum = compton_electron_spectrum(2.0, e_edges_test)
norm = np.sum(spectrum)
print(f"\nCompton recoil electron spectrum at 2 MeV ({len(spectrum)} bins):")
print(spectrum/norm)
print(f"Sum: {norm:.6e}")

# Also note that the minimum T is 0 (at θ=0) and max T = E*2*alpha/(1+2*alpha)
alpha_2 = 2.0/mec2
T_max = 2.0 * (2*alpha_2/(1+2*alpha_2))
print(f"Max electron energy from Compton at 2 MeV: {T_max:.4f} MeV")