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
    """dσ/dΩ per electron (cm^2/sr), Klein-Nishina. Returns (dsigma, E_scattered)"""
    alpha = E / mec2
    k = 1.0 / (1.0 + alpha * (1.0 - costheta))
    Ep = E * k
    return (r0_sq/2.0) * k**2 * (k + 1.0/k - (1.0 - costheta**2)), Ep

def sigma_C(E):
    """Total Compton cross section per electron (cm^2)"""
    ct = np.linspace(-1, 1, 10000)
    f, _ = dsigma_dOmega(E, ct)
    return 2*np.pi * integrate.simpson(f, ct)

print(f"sigma_C(2 MeV) via integration: {sigma_C(2.0):.6e} cm^2")

# Average energy transfer
def compton_energy_quantities(E):
    costheta = np.linspace(-1, 1, 100000)
    ds, Ep = dsigma_dOmega(E, costheta)
    dfdx = 2*np.pi * ds  # dσ/d(cosθ)
    sigma = integrate.simpson(dfdx, costheta)
    avg_Ep = integrate.simpson(dfdx * Ep, costheta) / sigma
    avg_E_transfer = E - avg_Ep
    return sigma, avg_Ep, avg_E_transfer

sigma, avg_Ep, avg_ET = compton_energy_quantities(2.0)
print(f"\nCompton at 2 MeV:")
print(f"  sigma per electron: {sigma:.6e} cm^2")
print(f"  Average scattered photon energy: {avg_Ep:.4f} MeV")
print(f"  Average energy transferred to electron: {avg_ET:.4f} MeV")
print(f"  Fraction energy transferred: {avg_ET/2.0:.4f}")

# Electron spectrum from Compton
def compton_electron_spectrum(E_photon, e_edges):
    costheta = np.linspace(-1, 1, 200000)
    ds, Ep = dsigma_dOmega(E_photon, costheta)
    T = E_photon - Ep
    counts, _ = np.histogram(T, bins=e_edges, weights=2*np.pi*ds)
    return counts

e_edges_25 = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.4])
spectrum = compton_electron_spectrum(2.0, e_edges_25)
norm = np.sum(spectrum)
print(f"\nCompton recoil electron spectrum at 2 MeV normalized:")
print(np.round(spectrum/norm, 5))
print(f"Sum of weights: {norm:.6e}")

alpha_2 = 2.0/mec2
T_max = 2.0 * (2*alpha_2/(1+2*alpha_2))
print(f"\nMax electron energy from Compton at 2 MeV: {T_max:.4f} MeV")

# Also compute the angular distribution of recoil electrons
def compton_electron_angular(E, ct_edges):
    costheta_ph = np.linspace(-1, 1, 200000)
    ds, Ep = dsigma_dOmega(E, costheta_ph)
    E_ph = E
    cos_e = (E_ph - Ep*costheta_ph) / np.sqrt(E_ph**2 + Ep**2 - 2*E_ph*Ep*costheta_ph)
    counts, _ = np.histogram(cos_e, bins=ct_edges, weights=2*np.pi*ds)
    return counts

ct_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
ang = compton_electron_angular(2.0, ct_edges)
print(f"\nCompton electron angular distribution ({len(ang)} bins):")
print(np.round(ang/np.sum(ang), 5))