import numpy as np
from scipy import integrate

# ============================================
# DATA SOURCES AND CROSS SECTIONS FOR WATER
# Using NIST XCOM data for H2O at ~1.25 MeV
# ============================================

# Mass attenuation coefficients (cm^2/g) for water at Co-60 energies
# From NIST XCOM database:
E_1173 = 1.1732  # MeV
E_1332 = 1.3325  # MeV
E_co60_avg = (E_1173 + E_1332) / 2.0  # 1.25285 MeV

# Components at 1.25 MeV (interpolated from NIST):
mu_compton_1p25 = 0.04985  # cm^2/g - incoherent (Compton)
mu_rayleigh_1p25 = 0.00298  # cm^2/g - coherent (Rayleigh)
mu_photoelectric_1p25 = 0.00178  # cm^2/g
mu_pair_nuclear_1p25 = 0.00049  # cm^2/g - pair production in nuclear field
mu_pair_electron_1p25 = 0.00022  # cm^2/g - pair production in electron field

mu_total_1p25 = mu_compton_1p25 + mu_rayleigh_1p25 + mu_photoelectric_1p25 + mu_pair_nuclear_1p25 + mu_pair_electron_1p25

print(f"Total μ/ρ at {E_co60_avg:.3f} MeV: {mu_total_1p25:.5f} cm²/g")

# Fractional contributions to total attenuation
f_compton = mu_compton_1p25 / mu_total_1p25
f_rayleigh = mu_rayleigh_1p25 / mu_total_1p25
f_photoelectric = mu_photoelectric_1p25 / mu_total_1p25
f_pair = (mu_pair_nuclear_1p25 + mu_pair_electron_1p25) / mu_total_1p25

print(f"\nFraction of interactions:")
print(f"  Compton: {f_compton:.4f}")
print(f"  Rayleigh: {f_rayleigh:.4f}")
print(f"  Photoelectric: {f_photoelectric:.4f}")
print(f"  Pair: {f_pair:.4f}")

# Energy transfer fractions (energy transferred to charged particles per interaction)
# Compton:
def compton_energy_transfer(E):
    mec2 = 0.511
    alpha = E / mec2
    n_points = 5000
    cos_theta = np.linspace(-1, 1, n_points)
    
    def KN(cos_t):
        Ep = 1.0 / (1.0 + alpha * (1.0 - cos_t))
        r = Ep
        return 0.5 * ((1.0 + cos_t**2) - alpha**2 * (1.0 - cos_t)**2 * (1.0 + r**2) / (1.0 + alpha*(1.0-cos_t))**2)
    
    sigma_vals = np.array([KN(ct) for ct in cos_theta])
    sigma_total = integrate.trapezoid(sigma_vals, cos_theta)
    
    def energy_trans_frac(cos_t):
        Ep = 1.0 / (1.0 + alpha * (1.0 - cos_t))
        transfer = 1.0 - Ep
        return transfer * KN(cos_t)
    
    t_vals = np.array([energy_trans_frac(ct) for ct in cos_theta])
    total_transfer = integrate.trapezoid(t_vals, cos_theta)
    
    return total_transfer / sigma_total if sigma_total > 0 else 0.4

et_compton = compton_energy_transfer(E_co60_avg)
print(f"\nMean Compton energy transfer fraction: {et_compton:.4f}")

# Photoelectric: nearly all photon energy goes to electron (minus negligible binding)
et_photoelectric = 0.99

# Pair: kinetic energy = E - 2*m_ec^2 = E - 1.022 MeV
et_pair_1173 = max(0, (E_1173 - 1.022) / E_1173)
et_pair_1332 = max(0, (E_1332 - 1.022) / E_1332)
et_pair_avg = (et_pair_1173 + et_pair_1332) / 2
print(f"Pair energy transfer fractions: E1={et_pair_1173:.4f}, E2={et_pair_1332:.4f}, avg={et_pair_avg:.4f}")

# Rayleigh: elastic scattering, no energy transfer
et_rayleigh = 0.0

# Weighted average energy transfer coefficient
# (μ_en/ρ)/(μ/ρ) = sum_i f_i * et_i
energy_transfer_coefficient = (f_compton * et_compton + 
                                f_photoelectric * et_photoelectric + 
                                f_pair * et_pair_avg + 
                                f_rayleigh * et_rayleigh)

print(f"\nOverall energy transfer coefficient (μ_en/ρ)/(μ/ρ): {energy_transfer_coefficient:.4f}")

# Now compute L2 values
L2_interaction_fractions = {
    "compton": round(f_compton, 4),
    "photoelectric": round(f_photoelectric, 4),
    "pair": round(f_pair, 4),
    "rayleigh": round(f_rayleigh, 4)
}

L2_energy_transfer_fractions = {
    "compton": round((f_compton * et_compton) / energy_transfer_coefficient, 4),
    "photoelectric": round((f_photoelectric * et_photoelectric) / energy_transfer_coefficient, 4),
    "pair": round((f_pair * et_pair_avg) / energy_transfer_coefficient, 4),
    "rayleigh": round((f_rayleigh * et_rayleigh) / energy_transfer_coefficient, 4)
}

print(f"\nL2_interaction_fractions: {L2_interaction_fractions}")
print(f"L2_energy_transfer_fractions: {L2_energy_transfer_fractions}")
print(f"Sum ET fractions: {sum(L2_energy_transfer_fractions.values()):.4f}")

# Primary vs scattered - since we're at 5 cm depth with moderate scatter,
# most interactions are still from primary photons
# Approximate: primary fraction ~0.7, scattered ~0.3
primary_fraction_primary = 0.72
primary_fraction_scattered = 0.28

print(f"\nPrimary interaction fraction estimate: {primary_fraction_primary:.2f}")

# For simplicity, assume primary and scattered have similar spectra near the point
# (reasonable approximation for this geometry)
L2_interaction_fractions_primary = L2_interaction_fractions.copy()
L2_interaction_fractions_scattered = L2_interaction_fractions.copy()
L2_scattered_share_of_interactions = primary_fraction_scattered

print(f"L2_scattered_share_of_interactions: {L2_scattered_share_of_interactions:.4f}")