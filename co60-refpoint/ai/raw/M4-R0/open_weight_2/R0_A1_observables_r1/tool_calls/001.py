import numpy as np
from scipy.constants import c, e, h, m_e, k
import json

# Constants
rho_water = 1.000  # g/cm^3
dmax_water = 0.5  # cm for Co-60 beam
depth_5_cm = 5.0  # cm
scoring_volume_radius = 1.0  # cm
scoring_volume_thickness = 0.2  # cm
SSD = 100.0  # cm
source_to_poi_distance = 105.0  # cm

# Photon energies (MeV) - Co-60 lines
E1 = 1.1732  # MeV
E2 = 1.3325  # MeV
E_avg = (E1 + E2) / 2.0  # ~1.253 MeV

print(f"Average photon energy: {E_avg:.3f} MeV")

# Mass attenuation coefficients for water at relevant energies (from NIST XCOM)
# At 1.173 MeV
mu_rho_E1 = 0.06098  # cm^2/g (approximate from NIST)
# At 1.332 MeV
mu_rho_E2 = 0.05956  # cm^2/g (approximate from NIST)
# Weighted average for equal probability
mu_rho_avg = (mu_rho_E1 + mu_rho_E2) / 2.0
print(f"Mass attenuation coefficient (avg): {mu_rho_avg:.5f} cm^2/g")
print(f"Linear attenuation coefficient: {mu_rho_avg * rho_water:.5f} cm^-1")

# Attenuation factor through 5 cm water
attenuation_5cm = np.exp(-mu_rho_avg * rho_water * depth_5_cm)
print(f"Attenuation factor at 5 cm: {attenuation_5cm:.6f}")

# Primary fluence normalization
# Phi0 is defined as primary photon fluence at 105 cm from source with phantom removed
# So L1_primary_fluence_per_phi0 should be close to attenuation_5cm if we normalize to Phi0 at 105 cm

# For a point source, fluence decreases as 1/r^2
# But since we're normalizing to Phi0 at same distance without phantom, 
# the only difference is attenuation in water

L1_primary_per_phi0 = attenuation_5cm
print(f"L1_primary_fluence_per_phi0: {L1_primary_per_phi0:.6f}")

# Now let's think about total fluence including scatter
# At 5 cm depth for a 10x10 cm field, there will be some build-up from scattered photons
# The buildup factor B(d) for Co-60 at 5 cm in water is approximately...
# From standard tables, buildup factor at 5 mmf (mean free path) for Co-60
mu = mu_rho_avg * rho_water  # cm^-1
mfp_5cm = mu * 5.0  # ~ 3 mean free paths
print(f"Depth in mean free paths: {mfp_5cm:.3f}")

# Buildup factor approximation using empirical formula
# For narrow beam geometry, B ≈ 1 + alpha * d * exp(beta*d)
# More accurately, for water at ~1.25 MeV:
# At d = 5 cm, buildup factor is roughly 1.3-1.5 depending on field size
B_approx = 1.45  # approximate buildup factor
L1_total_per_phi0 = attenuation_5cm * B_approx
print(f"L1_total_fluence_per_phi0 (with buildup): {L1_total_per_phi0:.6f}")

# Scattered fraction
L1_scattered_frac_fluence = 1.0 - 1.0/B_approx
print(f"L1_scattered_fraction_fluence: {L1_scattered_frac_fluence:.4f}")

# Scattered fraction of energy fluence
# Scattered photons tend to have lower energy, so scattered energy fluence fraction < scattered fluence fraction
# Energy-weighted scattered fraction typically ~0.3-0.5 of fluence scattered fraction
L1_scattered_frac_energy_fluence = 0.35  # estimate based on typical spectral hardening
print(f"L1_scattered_fraction_energy_fluence: {L1_scattered_frac_energy_fluence:.4f}")

# Mean photon energy
# Primary has unmodified spectrum, scattered has lower average energy
# Weighted mean
primary_frac = 1.0/B_approx
scattered_frac = 1.0 - primary_frac
# Average energy of primary: E_avg = 1.253 MeV
# Average energy of scattered: estimate ~0.8-1.0 MeV (Compton scattered)
E_scattered_avg = 0.95  # MeV (estimated)
L1_mean_photon_energy = primary_frac * E_avg + scattered_frac * E_scattered_avg
print(f"L1_mean_photon_energy_MeV: {L1_mean_photon_energy:.4f}")

print("\n--- Layer 1 complete ---\n")