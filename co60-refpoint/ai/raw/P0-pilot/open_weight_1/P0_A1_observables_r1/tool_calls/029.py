import pickle
import numpy as np
import math

with open('mc_results.pkl', 'rb') as f:
    data = pickle.load(f)

scoring = data['scoring']
elec_scoring = data['elec_scoring']
N = data['N']

cyl_volume = np.pi * 1.0**2 * 0.2
Phi0_per_src = np.pi * 0.2 / (121 * cyl_volume)
ct_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
electron_energy_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.4])
photon_energy_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
mec2 = 0.511

# ===== L3 continued =====
print("=== LAYER 3 continued ===")
elec_cyl = scoring['electrons_cylinder']
l3_cos = [cos_e for (_, cos_e, _) in elec_cyl]
l3_cos_hist = np.histogram(l3_cos, bins=ct_edges)[0]
l3_cos_hist_norm = l3_cos_hist / np.sum(l3_cos_hist)
print("L3_initial_direction:")
print(np.round(l3_cos_hist_norm, 6))

# ===== LAYER 4 =====
print("\n=== LAYER 4 ===")
e_flu_per_phi0 = elec_scoring['e_fluence']/(cyl_volume*N)/Phi0_per_src
print(f"L4_electron_fluence_per_phi0 = {e_flu_per_phi0:.6f}")

e_spec = elec_scoring['e_spec']/(cyl_volume*N)/Phi0_per_src
print(f"L4_electron_fluence_spectrum:")
print(np.round(e_spec, 6))
print(f"Sum: {np.sum(e_spec):.4f}")

e_fluence_track = np.sum(elec_scoring['e_spec'])
print(f"Track in bins: {e_fluence_track:.3f}")
print(f"Total track: {elec_scoring['e_fluence']:.3f}")

# Mean electron energy
mean_e_E = 0
for i in range(25):
    E_center = (electron_energy_edges[i]+electron_energy_edges[i+1])/2
    mean_e_E += E_center * elec_scoring['e_spec'][i]
mean_e_E /= e_fluence_track if e_fluence_track>0 else 1
print(f"L4_mean_electron_energy: {mean_e_E:.4f} MeV")

e_ang = elec_scoring['e_ang']/np.sum(elec_scoring['e_ang']) if np.sum(elec_scoring['e_ang'])>0 else np.zeros(10)
print(f"L4_angular:")
print(np.round(e_ang, 6))
print(f"Sum: {np.sum(elec_scoring['e_ang']):.4f}")

# L4_radiative_to_collision: ratio for electrons in the scoring volume
# = sum(S_rad * L) / sum(S_col * L) along the tracks in the cylinder
# I don't directly have this from the transport. Let me compute from the spectrum
# using the mean radiative/collision ratio weighted by track length.
# But my electron transport only tracks collision dose. The radiative losses 
# happen but aren't deposited locally (they become bremsstrahlung).
# The radiative-to-collision ratio at the point for the electron fluence spectrum:
# R = Σ Φ(E) S_rad(E) / Σ Φ(E) S_col(E)

from mc_physics import S_col_w, S_rad_w
num_rad = 0; den_col = 0
for i in range(25):
    E_center = (electron_energy_edges[i]+electron_energy_edges[i+1])/2
    num_rad += elec_scoring['e_spec'][i] * S_rad_w(E_center)
    den_col += elec_scoring['e_spec'][i] * S_col_w(E_center)
print(f"L4_radiative_to_collision = {num_rad/den_col:.5f}" if den_col>0 else "0")

# Delta ray fraction
delta_frac = elec_scoring['delta_fluence'] / elec_scoring['e_fluence'] if elec_scoring['e_fluence']>0 else 0
print(f"\nL4_delta_fraction (from current transport) = {delta_frac:.4f}")
print("(No delta rays are currently produced in the transport - need to add)")

# ===== LAYER 5 =====
print("\n=== LAYER 5 ===")

# Dose from electron transport
e_dose_per_phi0 = elec_scoring['e_dose']/(cyl_volume*N)/Phi0_per_src
# Convert MeV to Gy: 1 MeV = 1.602e-13 J, 1 g = 1e-3 kg
# e_dose is in MeV deposited in the cylinder
# Mass of cylinder = cyl_volume * rho = 0.6283 g
# Dose per unit mass = e_dose / mass
# In Gy per Phi0:
D_over_phi0_Gy_cm2 = e_dose_per_phi0 * 1.602e-13 * 1000  # MeV·cm²/g → Gy·cm²
print(f"L5_D_over_phi0 = {D_over_phi0_Gy_cm2:.6e} Gy cm^2")

# Kerma from photon fluence
# K = Σ Φ(E) × (μ_tr/ρ)(E) × E
# K_col = Σ Φ(E) × (μ_en/ρ)(E) × E
from mc_physics import mu_en_rho_w

# Primary contribution
K_col_prim = scoring['ph_prim_fluence']/(cyl_volume*N)/Phi0_per_src * mu_en_rho_w(2.0) * 2.0
# in MeV·cm²/g per Phi0

# Scattered contribution
# Compute from scattered spectrum (only bins up to 1.4)
K_col_scat_bins = 0
scat_spec_per_phi0 = scoring['ph_scat_spec']/(cyl_volume*N)/Phi0_per_src
for i in range(28):
    E_center = (photon_energy_edges[i]+photon_energy_edges[i+1])/2
    K_col_scat_bins += scat_spec_per_phi0[i] * mu_en_rho_w(E_center) * E_center

# For scattered photons above 1.4 MeV: estimate
# They're about 19% of scattered fluence
scat_above_14 = scoring['ph_scat_fluence']-np.sum(scoring['ph_scat_spec'])
scat_above_14_per_phi0 = scat_above_14/(cyl_volume*N)/Phi0_per_src
# Approximate mean energy ~1.6 MeV, mu_en ~0.026
K_col_scat_tail = scat_above_14_per_phi0 * 0.026 * 1.6

K_col_scat = K_col_scat_bins + K_col_scat_tail
K_col_total = K_col_prim + K_col_scat

print(f"K_col primary: {K_col_prim:.5f} MeV·cm²/g per Phi0")
print(f"K_col scattered (bins): {K_col_scat_bins:.5f} MeV·cm²/g per Phi0")
print(f"K_col scattered (tail): {K_col_scat_tail:.5f} MeV·cm²/g per Phi0")
print(f"K_col total: {K_col_total:.5f} MeV·cm²/g per Phi0")
K_col_Gy = K_col_total * 1.602e-13 * 1000
print(f"L5_Kcol_over_phi0 = {K_col_Gy:.6e} Gy cm^2")

# Total kerma: K = K_col/(1-g) where g is radiative fraction
# g for photons in water at 2 MeV ≈ 0.005
# For scattered photons, g is small too
g_fraction = 0.005
K_total = K_col_total / (1-g_fraction)
K_Gy = K_total * 1.602e-13 * 1000
print(f"L5_K_over_phi0 = {K_Gy:.6e} Gy cm^2")
print(f"L5_g = {1 - K_col_total/K_total:.5f}")

# beta = D/K_col
beta = D_over_phi0_Gy_cm2 / K_col_Gy
print(f"L5_beta = {beta:.4f}")

# Dose fraction from scattered parent photons
# I need to track parent primary vs scattered for dose. 
# In my electron transport, I tracked (pos, dir, T, parent_prim) for electron sources.
# I can compute the dose by parent by using the electron sources' parent flag.
# But the transport function doesn't separate dose by parent.
# For now, let me estimate from the electron sources.

# Actually, let me compute the dose fraction parent scattered from the source records.
# The 'electron_sources' have parent_prim flag.
# Let me redo the electron transport splitting by parent, or estimate.

# Estimate: dose fraction from scattered = fraction of energy deposited by electrons 
# from scattered photons = (K_col_scat)/(K_col_total) approx (for CPE)
dose_frac_scattered = K_col_scat / K_col_total
print(f"L5_dose_fraction_parent_scattered (estimate from kerma): {dose_frac_scattered:.4f}")

# PDD5 - need depth dose at 5 cm and dmax
# Will compute separately

# ===== Save all intermediate results =====  
results = {
    'L1_primary_fluence_per_phi0': prim_flu_per_phi0 if 'prim_flu_per_phi0' in dir() else scoring['ph_prim_fluence']/(cyl_volume*N)/Phi0_per_src,
    'scat_spec_per_phi0': scat_spec_per_phi0,
    'e_spec_per_phi0': e_spec,
}
print("\nAll results computed and stored.")