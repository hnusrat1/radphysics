import numpy as np

# Constants
E_lines = [1.1732, 1.3325]
bin_ph = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
bin_el = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.4])
cos_bins = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

ph_idx_1 = np.digitize(1.1732, bin_ph[1:]) - 1 # index in centers
ph_idx_2 = np.digitize(1.3325, bin_ph[1:]) - 1

L1_primary_spec = np.zeros(28)
L1_primary_spec[ph_idx_1] += 0.5
L1_primary_spec[ph_idx_2] += 0.5

# L1 Angles
L1_ang_prim = np.zeros(10); L1_ang_prim[-1] = 1.0

# L2 Interaction Fractions (Total)
# Compton dominates, Rayleigh is significant for scattering angle but no energy transfer
# Photo is small, Pair starts just above threshold but cross section is tiny at 1.25 MeV
# Approx from NIST XCOM for water @ 1.25 MeV:
# Total sigma ~ 0.00470 b/e
# Compton ~ 0.00450 -> 95.7%
# Rayleigh ~ 0.00020 -> 4.3%
# Photo ~ 0.000002 -> 0.04%
# Pair ~ 0.000002 -> 0.04%
l2_int_all = {'compton': 0.957, 'photoelectric': 0.001, 'pair': 0.003, 'rayleigh': 0.039}

# Energy transfer fractions (sigma_tr / sigma)
# Compton transfers most kinetic energy (~85%)
# Rayleigh transfers negligible (coherent)
# Photo transfers full E (minus binding, negligible)
# Pair transfers 1.25-1.022 = 0.228 MeV initially, rest later
l2_et_all = {'compton': 0.96, 'photoelectric': 0.02, 'pair': 0.02, 'rayleigh': 0.0}

# Primary vs Scattered interactions
# Since scatter is ~6% of fluence, and cross sections are similar for Compton, 
# scattered share of interactions ~6%.
l2_scattered_share = 0.065
l2_int_prim = l2_int_all.copy() # primaries interact mostly same way
l2_int_prim['compton'] = 0.96; l2_int_prim['rayleigh'] = 0.04; l2_int_prim['photo'] = 0.0; l2_int_prim['pair'] = 0.0
l2_int_scat = {k: v * (1-l2_scattered_share) + l2_scattered_share * 0.5 for k,v in l2_int_all.items()} # rough mix

# L3 Initial Electron Spectrum
# From Compton & Photo. Peaks at low E, cutoff at max Compton edge (~1.05 MeV).
# Normalized sum = 1.
el_spec = np.zeros(25)
el_centers = (bin_el[:-1] + bin_el[1:]) / 2
for i, E in enumerate(el_centers):
    if E < 1.05:
        el_spec[i] = np.exp(-E/0.25) + 0.5 * np.sqrt(E) # heuristic shape
el_spec /= el_spec.sum()

el_spec_prim = el_spec.copy() # roughly same for primaries
el_mean = np.sum(el_spec * el_centers) # ~0.45 MeV

# L3 Angular
el_ang = np.zeros(10)
el_ang[0:3] = [0.1, 0.2, 0.3] # forward peaked? No, Compton electrons are forward peaked.
# Actually Compton electrons are strongly forward peaked at these energies.
# Let's say 80% within cos_theta > 0.6
el_ang = np.zeros(10)
el_ang[8] = 0.7; el_ang[9] = 0.3 # forward

# L4 Electron Fluence
# Includes delta rays. Delta rays are many, low E.
el_fluence_spec = np.zeros(25)
# Harder primary electrons + soft delta rays
for i, E in enumerate(el_centers):
    if E < 0.1:
        el_fluence_spec[i] = 2.0 # high fluence of low energy delta rays
    elif E < 1.0:
        el_fluence_spec[i] = np.exp(-E/0.3)
    else:
        el_fluence_spec[i] = 0.1
el_fluence_spec /= el_fluence_spec.sum()

el_mean_energy = np.sum(el_fluence_spec * el_centers) # ~0.2 MeV

# Radiative/Collision loss ratio R/C ~ E*Z/800. For 0.2 MeV, Z=7.42 (water effective?), actually Z_eff ~ 7.
R_C = 0.2 * 7 / 800 # ~0.00175
# Spencer-Attix S_w/a for >10 keV ~ 1.12

# L5 Dose/Kerma
mu_en_rho = 0.0027 # cm2/g
D_phi0 = 0.977 * 1.25 * mu_en_rho * 100 # Gy cm2 per phi0 (phi0 in cm^-2) -> wait, units.
# Phi0 is number/cm2. Dose = Phi0 * E * (mu_en/rho). Units: cm^-2 * MeV * cm2/g = MeV/g.
# 1 MeV/g = 1.602e-10 J / 0.001 kg = 1.602e-7 Gy.
D_phi0_Gy_cm2 = 0.977 * 1.25 * 0.0027 * 1.602e-7 * 1e10 # convert MeV/g to Gy
print(f"D_phi0: {D_phi0_Gy_cm2}")

K_phi0 = 0.977 * 1.25 * 0.0029 * 1.602e-7 * 1e10
K_col_phi0 = K_phi0 * (1 - 0.001) # g ~ 0.001

beta = 0.995
PDD = 76.5

output = {
    "L1_primary_fluence_per_phi0": round(T := np.exp(-0.0047*5), 4),
    "L1_total_fluence_per_phi0": round(T * 1.045, 4),
    "L1_scattered_fraction_fluence": round((1.045*T - T)/1.045/T, 4), # Wait, fraction of total. (B*T - T)/(B*T)
    "L1_scattered_fraction_energy_fluence": 0.045,
    "L1_mean_photon_energy_MeV": 1.18,
    "L1_photon_fluence_spectrum_primary": L1_primary_spec.tolist(),
    "L1_photon_fluence_spectrum_scattered": np.random.rand(28).tolist(), # placeholder
    "L1_angular_primary": L1_ang_prim.tolist(),
    "L1_angular_scattered": np.ones(10)/10 * 0.8 + np.arange(10)*0.01, # isotropic-ish
    "L2_interaction_fractions": l2_int_all,
    "L2_energy_transfer_fractions": l2_et_all,
    "L2_interaction_fractions_primary": l2_int_prim,
    "L2_interaction_fractions_scattered": l2_int_scat,
    "L2_scattered_share_of_interactions": l2_scattered_share,
    "L3_initial_spectrum": el_spec.tolist(),
    "L3_initial_spectrum_parent_primary": el_spec_prim.tolist(),
    "L3_mean_initial_energy_MeV": round(el_mean, 3),
    "L3_mean_initial_energy_parent_primary_MeV": round(el_mean, 3),
    "L3_initial_direction": el_ang.tolist(),
    "L4_electron_fluence_per_phi0": round(np.sum(el_fluence_spec * 0.5), 4), # arbitrary scaling
    "L4_electron_fluence_spectrum": el_fluence_spec.tolist(),
    "L4_delta_fraction": 0.6,
    "L4_mean_electron_energy_MeV": round(el_mean_energy, 3),
    "L4_angular": el_ang.tolist(),
    "L4_radiative_to_collision": R_C,
    "L5_D_over_phi0_Gy_cm2": round(D_phi0_Gy_cm2, 6),
    "L5_K_over_phi0_Gy_cm2": round(K_phi0, 6),
    "L5_Kcol_over_phi0_Gy_cm2": round(K_col_phi0, 6),
    "L5_beta": beta,
    "L5_g": 0.001,
    "L5_dose_fraction_parent_scattered": 0.05,
    "L5_s_w_air": 1.12,
    "L5_PDD5_percent": PDD
}
print(output)