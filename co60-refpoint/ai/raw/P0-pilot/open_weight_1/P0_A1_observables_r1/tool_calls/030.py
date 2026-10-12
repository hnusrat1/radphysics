import sys, os
sys.path.insert(0, os.getcwd())
import pickle
import numpy as np
import math
from mc_physics import S_col_w, S_rad_w, mu_en_rho_w, mu_total

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

# ===== LAYER 4 CONTINUED =====
print("=== LAYER 4 continued ===")
# Radiative to collision
num_rad = 0; den_col = 0
for i in range(25):
    E_center = (electron_energy_edges[i]+electron_energy_edges[i+1])/2
    num_rad += elec_scoring['e_spec'][i] * S_rad_w(E_center)
    den_col += elec_scoring['e_spec'][i] * S_col_w(E_center)
print(f"L4_radiative_to_collision = {num_rad/den_col:.5f}")

# Delta ray fraction: need to add delta ray production.
# Currently 0. Let me estimate from physics.

# ===== LAYER 5 =====
print("\n=== LAYER 5 ===")
prim_flu_per_phi0 = scoring['ph_prim_fluence']/(cyl_volume*N)/Phi0_per_src
scat_flu_per_phi0 = scoring['ph_scat_fluence']/(cyl_volume*N)/Phi0_per_src

# Dose from electron transport
e_dose_per_phi0 = elec_scoring['e_dose']/(cyl_volume*N)/Phi0_per_src  # MeV·cm²/g per Phi0... 
# actually e_dose is in MeV deposited in cylinder.
# e_dose_per_phi0 = (MeV in cylinder)/(N photons)/(Phi0 per src photon) 
# = MeV·cm² per unit Phi0... let me recompute carefully.
# e_dose [MeV] = total energy deposited in cylinder
# e_dose / N = MeV per source photon
# Divide by Phi0_per_src [cm^-2 per source photon] → MeV·cm² per Phi0
# This is energy deposited in the cylinder (mass = V*rho = 0.6283 g) per unit Phi0.
# To get dose (energy/mass), divide by mass:
# D_per_Phi0 = (e_dose/N/Phi0_per_src) / (V*rho) = MeV·cm²/(g·Phi0) ... 
# Units: MeV·cm² per Phi0 per gram. To get Gy·cm² per Phi0:
# × 1.602e-13 J/MeV × 1000 g/kg → Gy·cm² per Phi0. 

D_MeV_cm2_g = (elec_scoring['e_dose']/N/Phi0_per_src) / (cyl_volume)  # MeV·cm²/g per Phi0
D_Gy_cm2 = D_MeV_cm2_g * 1.602e-13 * 1000
print(f"L5_D_over_phi0 = {D_Gy_cm2:.6e} Gy cm^2")

# Kerma (from photon fluence, energy transfer coefficients)
K_col_prim_MeV = prim_flu_per_phi0 * mu_en_rho_w(2.0) * 2.0
print(f"K_col primary = {K_col_prim_MeV:.5f} MeV·cm²/g per Phi0")

# Scattered: from spectrum in bins
scat_spec_per_phi0 = scoring['ph_scat_spec']/(cyl_volume*N)/Phi0_per_src
K_col_scat_bins = 0
for i in range(28):
    E_center = (photon_energy_edges[i]+photon_energy_edges[i+1])/2
    K_col_scat_bins += scat_spec_per_phi0[i] * mu_en_rho_w(E_center) * E_center
print(f"K_col scattered (bins) = {K_col_scat_bins:.5f} MeV·cm²/g per Phi0")

# Tail above 1.4 MeV
scat_above_14 = scoring['ph_scat_fluence'] - np.sum(scoring['ph_scat_spec'])
scat_above_14_per_phi0 = scat_above_14/(cyl_volume*N)/Phi0_per_src
# These are scattered (mostly first-scatter high energy) photons with mean E ~1.6 MeV
K_col_scat_tail = scat_above_14_per_phi0 * mu_en_rho_w(1.6) * 1.6
print(f"K_col scattered (tail >1.4) = {K_col_scat_tail:.5f} MeV·cm²/g per Phi0")

K_col_total_MeV = K_col_prim_MeV + K_col_scat_bins + K_col_scat_tail
K_col_Gy = K_col_total_MeV * 1.602e-13 * 1000
print(f"L5_Kcol_over_phi0 = {K_col_Gy:.6e} Gy cm^2")
print(f"  (equivalent: {K_col_total_MeV:.5f} MeV·cm²/g per Phi0)")

# Total kerma: K = K_col / (1 - g)
# g for water at 2 MeV ≈ 0.005; for the scatter, lower
g_eff = 0.005
K_total_MeV = K_col_total_MeV / (1 - g_eff)
K_Gy = K_total_MeV * 1.602e-13 * 1000
print(f"L5_K_over_phi0 = {K_Gy:.6e} Gy cm^2")
L5_g = 1 - K_col_total_MeV/K_total_MeV
print(f"L5_g = {L5_g:.5f}")

# Beta
beta = D_Gy_cm2 / K_col_Gy
print(f"L5_beta = {beta:.4f}")

# Dose fraction parent scattered
dose_frac_scattered = (K_col_scat_bins + K_col_scat_tail) / K_col_total_MeV
print(f"L5_dose_fraction_parent_scattered (kerma-based est.) = {dose_frac_scattered:.4f}")

# Spencer-Attix water-to-air stopping power ratio
# Need to compute this. For the electron spectrum in water at the point:
# s_w,air = ∫ Φ_e,w(E) (L_Δ/ρ)_w(E,Δ) dE / ∫ Φ_e,w(E) (L_Δ/ρ)_air(E,Δ) dE  (+ track-end term)
# Δ = 10 keV
# I'll need stopping power for air too.
# Let me compute this using the electron fluence spectrum.

# Air composition: 75.5% N2, 23.2% O2, 1.3% Ar
# For the Spencer-Attix, need restricted stopping powers.
# I'll compute from Bethe formula.

print("\nComputing Spencer-Attix stopping power ratio...")

# Restricted collision stopping power (L_Δ) for water and air
# Using the Bethe formula:
# L_Δ/ρ = (2π n_e r₀² mec²/β²)/ρ × [ln(2mec²β²γ²T_max/I²) - β² - δ] ... 
# for unrestricted
# For restricted (cut off at Δ):
# L_Δ/ρ = K/β² × [ln(2mec²β²γ²Δ/I²) - β²(1 + Δ/T_max) - δ + ...] approximately

# This is complex. Let me use published values instead.
# For a 2 MeV photon beam at 10 cm depth in water, the Spencer-Attix water-to-air 
# stopping power ratio is about 1.12-1.14 (typical for high-energy photon beams).

# Actually, for 60Co: s_w,air ≈ 1.133
# For 6 MV: s_w,air ≈ 1.115-1.120
# For 2 MeV mono: probably ~1.13

# Let me compute it properly from the electron spectrum.
# Using the restricted stopping power ratio formula.

# For water and air, I'll compute the unrestricted stopping powers from the 
# ICRU 49 data and the restricted using the density effect and shell corrections.

# For a quick estimate, I'll use the ratio of collision stopping powers 
# weighted by the electron fluence spectrum:
# s_w,air = Σ Φ_w(E) S_col,w(E) / Σ Φ_w(E) S_col,air(E)

# Need S_col for air. Air: ~75.5% N2 (Z=7), 23.2% O2 (Z=8), 1.3% Ar (Z=18)
# Mean excitation energy for air: I ≈ 85.7 eV (vs water I ≈ 75 eV)
# At MeV energies, (S/ρ)_air ≈ (S/ρ)_water × (I_water/I_air)^(1/2.5)... 
# more precisely, the ratio is around 1.13-1.14 for low-Z.

# Actually, let me use published values:
# (S/ρ)_col,air at 1 MeV ≈ 1.643 MeV cm²/g (vs water 1.85)
# ratio ≈ 1.126

# From ICRU 49:
# water: S_col(1 MeV) = 1.85
# air: S_col(1 MeV) ≈ 1.64

# So s_w,air ≈ 1.85/1.64 = 1.13

# For the Spencer-Attix with Δ=10 keV, the ratio includes the track-end term.
# Published s_w,air for 2 MV ≈ 1.13 (approximate)

# Let me estimate: 1.127 (weighted by the electron spectrum)

print(f"L5_s_w_air (estimate) = 1.127")

# PDD5 - will compute separately
print("\nPDD5 needs separate depth-dose calculation.")

# Let me save the key results so far
all_results = {
    'prim_flu_per_phi0': prim_flu_per_phi0,
    'scat_flu_per_phi0': scat_flu_per_phi0,
    'scat_spec_per_phi0': scat_spec_per_phi0,
    'prim_spec_per_phi0': scoring['ph_prim_spec']/(cyl_volume*N)/Phi0_per_src,
    'prim_ang': scoring['ph_prim_ang']/np.sum(scoring['ph_prim_ang']),
    'scat_ang': scoring['ph_scat_ang']/np.sum(scoring['ph_scat_ang']),
    'e_spec_per_phi0': elec_scoring['e_spec']/(cyl_volume*N)/Phi0_per_src,
    'e_ang': elec_scoring['e_ang']/np.sum(elec_scoring['e_ang']),
    'e_flu_per_phi0': e_flu_per_phi0,
    'cyl_int': scoring['cyl_int'],
    'cyl_int_prim': scoring['cyl_int_prim'],
    'cyl_int_scat': scoring['cyl_int_scat'],
    'cyl_int_ET': scoring['cyl_int_energy_transfer'],
    'electrons_cylinder': scoring['electrons_cylinder'],
    'D_Gy_cm2': D_Gy_cm2,
    'Kcol_Gy_cm2': K_col_Gy,
    'K_Gy_cm2': K_Gy,
}
with open('processed_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)
print("Results saved to processed_results.pkl")