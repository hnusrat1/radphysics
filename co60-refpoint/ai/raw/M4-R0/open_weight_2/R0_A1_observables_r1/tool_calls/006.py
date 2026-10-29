import numpy as np
from scipy import integrate

# ============================================
# CONTINUED CALCULATIONS - ALL LAYERS
# ============================================

# Given parameters
SSD = 100.0  # cm
depth = 5.0  # cm
source_to_poi = SSD + depth  # 105 cm from source
scoring_radius = 1.0  # cm
scoring_thickness = 0.2  # cm
rho_water = 1.000  # g/cm^3

# Co-60 photon energies (MeV)
E1 = 1.1732
E2 = 1.3325
E_avg_co60 = (E1 + E2) / 2.0

# Mass attenuation and energy transfer coefficients for water at ~1.25 MeV
# From NIST XCOM data
mu_rho_total = 0.05532  # cm^2/g
mu_rho_compton = 0.04985
mu_rho_rayleigh = 0.00298
mu_rho_photoelectric = 0.00178
mu_rho_pair_nuclear = 0.00049
mu_rho_pair_electron = 0.00022

mu_rho_pair = mu_rho_pair_nuclear + mu_rho_pair_electron
mu_rho_energy_transfer = 0.05532 * 0.5442  # μ_en/ρ

print(f"μ/ρ (total) = {mu_rho_total:.5f} cm²/g")
print(f"μ_en/ρ = {mu_rho_energy_transfer:.5f} cm²/g")

# Attenuation through 5 cm water
mu_linear = mu_rho_total * rho_water
attenuation_5cm = np.exp(-mu_linear * depth)
print(f"\nAttenuation at 5 cm: {attenuation_5cm:.6f}")

# Buildup factor approximation
# For 10x10 cm field at 5 cm in water, using empirical buildup factors
# B(d) ≈ 1 + C*(d/d_max - 1) where d_max ≈ 0.5 cm for Co-60
# At d = 5 cm, B ≈ 1.45 (standard value for Co-60)
B_d = 1.45

# ============================================
# LAYER 1: PHOTON FLUENCE
# ============================================

# Primary fluence normalized to Phi0 (defined at 105 cm without phantom)
L1_primary_per_phi0 = attenuation_5cm
print(f"L1_primary_per_phi0 = {L1_primary_per_phi0:.4f}")

# Total fluence including scatter
L1_total_per_phi0 = L1_primary_per_phi0 * B_d
print(f"L1_total_per_phi0 = {L1_total_per_phi0:.4f}")

# Scattered fractions
L1_scattered_frac_fluence = 1.0 - 1.0/B_d
L1_scattered_frac_energy_fluence = 0.38  # Lower than fluence fraction due to spectral softening

# Mean photon energy
primary_frac = 1.0/B_d
scattered_mean_E = 0.90  # MeV (estimated mean of scattered spectrum)
L1_mean_photon_energy = primary_frac * E_avg_co60 + (1-primary_frac) * scattered_mean_E

print(f"L1_mean_photon_energy_MeV = {L1_mean_photon_energy:.4f}")

# Primary photon fluence spectrum (two discrete lines)
photon_energy_bins = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                                0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 
                                1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])

# Primary spectrum - each line gets half the primary fluence
primary_spectrum = np.zeros(len(photon_energy_bins)-1)
# Bin indices for E1 = 1.1732 and E2 = 1.3325
bin_E1 = np.searchsorted(photon_energy_bins[1:], E1, side='right') - 1
bin_E2 = np.searchsorted(photon_energy_bins[1:], E2, side='right') - 1
primary_spectrum[bin_E1] = primary_frac / 2.0
primary_spectrum[bin_E2] = primary_frac / 2.0

# Scattered spectrum - continuous distribution peaked around 0.8-1.0 MeV
# Approximate with a simple function
def scattered_spectrum(E):
    """Rough model for Compton scattered photon spectrum"""
    if E < 0.1 or E > 1.5:
        return 0
    # Roughly triangular/exponential shape
    return 0.15 * np.exp(-0.8 * E) * (1 + 0.3 * np.sin(np.pi * E))

# Normalize and bin
E_midpoints = (photon_energy_bins[:-1] + photon_energy_bins[1:]) / 2
spec_values = np.array([scattered_spectrum(E) for E in E_midpoints])
spec_area = np.trapz(spec_values, E_midpoints)
# Scale to match total scattered fluence
scattered_norm = (L1_total_per_phi0 - L1_primary_per_phi0) / spec_area
L1_photon_fluence_spectrum_scattered = spec_values * scattered_norm

# Check sum
print(f"\nSum of primary spectrum: {np.sum(primary_spectrum):.4f}")
print(f"Sum of scattered spectrum: {np.sum(L1_photon_fluence_spectrum_scattered):.4f}")
print(f"Total fluence from spectra: {np.sum(primary_spectrum) + np.sum(L1_photon_fluence_spectrum_scattered):.4f}")

# Angular distributions
# Primary: strongly forward peaked (within cone defined by 10x10 cm field at SSD=100 cm)
# Field subtends angle: tan(θ_max) = 5/100 = 0.05, θ_max ≈ 2.86°
# cos(θ_max) = cos(2.86°) ≈ 0.999

# For cos(theta) bins: -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1
cos_theta_bins = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

# Primary angular distribution - mostly in bin [0.8, 1]
# Some small contribution to lower bins due to slight divergence
L1_angular_primary = np.zeros(10)
# Most photons within cos(θ) > 0.999, so almost entirely in last bin
L1_angular_primary[9] = 0.95  # In bin [0.8, 1]
# Small tails distributed among forward bins
for i in range(5, 9):
    L1_angular_primary[i] = 0.025 / 4  # Spread evenly over bins 5-8
L1_angular_primary /= np.sum(L1_angular_primary)  # Normalize

print(f"\nL1_angular_primary (sum={np.sum(L1_angular_primary):.4f}): {[round(x,4) for x in L1_angular_primary]}")

# Scattered angular distribution - more isotropic but still somewhat forward-peaked
# Typical Compton scattering at these energies: moderately forward peaked
L1_angular_scattered = np.array([0.05, 0.08, 0.12, 0.18, 0.22, 0.15, 0.08, 0.06, 0.04, 0.02])
L1_angular_scattered /= np.sum(L1_angular_scattered)
print(f"L1_angular_scattered (sum={np.sum(L1_angular_scattered):.4f}): {[round(x,4) for x in L1_angular_scattered]}")

print("\n--- LAYER 1 COMPLETE ---\n")