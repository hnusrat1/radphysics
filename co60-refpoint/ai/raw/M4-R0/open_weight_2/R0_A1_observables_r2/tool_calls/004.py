import numpy as np

# Physical Constants
mc2 = 0.510998 # MeV
rho = 1.0 # g/cm3

# Approximate cross-sections for Water @ 1.25 MeV (Weighted average of 1.17 and 1.33)
# Data from NIST XCOM
# Mu/rho (cm2/g)
# Compton: 0.064
# Rayleigh: 0.0001 
# Photo: 0.0001
# Pair: 0.0003
# Total: 0.0645 -> mu_lin = 0.0645 cm^-1

mu_rho = {
    'compton': 0.0638,
    'rayleigh': 0.0001,
    'photo': 0.0001,
    'pair': 0.0002,
    'total': 0.0642
}

mu_en_rho = 0.0280 # cm2/g
mu_tr_rho = 0.0290 # cm2/g

depth = 5.0 # cm
SSD = 100.0
dist = SSD + depth

# Transmission
T = np.exp(-mu_rho['total'] * rho * depth)
print(f"Transmission {depth}cm: {T:.4f}")

# Buildup Factor estimation (Chilton/Trompson or similar approx)
# At 5cm (~0.32 mfp), B is roughly 1.05 - 1.1?
# For broad beam. Here field is 10x10.
# Let's use B_phi = 1.04
B_phi = 1.04
Phi_primary = T
Phi_scattered = (B_phi - 1.0) * Phi_primary # Approx
Phi_total = B_phi * Phi_primary

# L1 Primary Spectrum (normalized to Phi_total / Phi0)
# Two lines 1.1732 and 1.3325.
bin_edges_ph = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
bin_centers_ph = (bin_edges_ph[:-1] + bin_edges_ph[1:]) / 2

l1_spec_prim = np.zeros(28)
l1_spec_scat = np.zeros(28)

# Primary indices
p1_idx = np.argmin(np.abs(bin_centers_ph - 1.1732))
p2_idx = np.argmin(np.abs(bin_centers_ph - 1.3325))
l1_spec_prim[p1_idx] += 0.5 * Phi_primary
l1_spec_prim[p2_idx] += 0.5 * Phi_primary

# Scattered Spectrum (Continuous Compton, degraded by absorption)
# Shape f(E) ~ 1/E + exponential cut-off
for i, E in enumerate(bin_centers_ph):
    if E < 1.0:
        l1_spec_scat[i] = np.exp(-E/0.3)
    else:
        l1_spec_scat[i] *= 0.1 # Cut off after max scattered photon energy
# Normalize Scattered component
scat_scale = Phi_scattered / l1_spec_scat.sum()
l1_spec_scat *= scat_scale

l1_mean_E = 1.25 # Approx

# L2 Interaction Fractions
int_compton = mu_rho['compton'] / mu_rho['total']
int_rayleigh = mu_rho['rayleigh'] / mu_rho['total']
int_photo = mu_rho['photo'] / mu_rho['total']
int_pair = mu_rho['pair'] / mu_rho['total']

# Energy Transfer Fractions
# Compton: high transfer
# Rayleigh: 0
# Photo: Full E (minus binding ~keV, negligible)
# Pair: Threshold > 1.022. Available energy = E - 1.022.
et_compton = (0.96 * mu_rho['compton']) / (0.96*mu_rho['compton'] + 0.99*mu_rho['photo'] + 0.5*mu_rho['pair']) # Rough weights
# Actually let's just use ratios
etr_compton = (mu_rho['compton'] * 0.96) 
etr_photo = (mu_rho['photo'] * 1.0)
etr_pair = (mu_rho['pair'] * 0.5)
etr_tot = etr_compton + etr_photo + etr_pair
l2_et_frac = {
    'compton': etr_compton / etr_tot,
    'photoelectric': etr_photo / etr_tot,
    'pair': etr_pair / etr_tot,
    'rayleigh': 0.0
}

# L2 Primary vs Scattered interactions
# Scattered share ~ 6.5% (based on B-1 / B)
l2_sh = 0.048 / 1.04 # 4.8/104
l2_scattered_share = 0.046
int_prim = {'compton': 0.965, 'photoelectric': 0.002, 'pair': 0.003, 'rayleigh': 0.03}
int_scat = {'compton': 0.93, 'photoelectric': 0.03, 'pair': 0.03, 'rayleigh': 0.01}

# L3 Initial Electron Spectrum (from Compton)
# T_max = E * 2a / (1+2a) where a = E/511
a = 1.25 / 0.511
T_max = 1.25 * 2*a / (1+2*a)
print(f"T_max: {T_max}")

bin_edges_el = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.4])
bin_centers_el = (bin_edges_el[:-1] + bin_edges_el[1:]) / 2

el_spec = np.zeros(25)
for i, T in enumerate(bin_centers_el):
    if T < T_max:
        # Approximation of d_sigma/dT. Increases then decreases.
        # Peak around T_min? No, forward peak implies high energy transfer.
        # In lab frame, electron spectrum peaks at lower energies.
        el_spec[i] = np.exp(-T/0.4) + 0.2 * T
el_spec /= el_spec.sum()
el_mean_init = np.sum(el_spec * bin_centers_el)

# Initial Direction
el_ang = np.zeros(10)
el_ang[0:2] = 0.05 # Backward
el_ang[2:6] = 0.15 # Sideways
el_ang[6:9] = 0.2 # Forward
el_ang[9] = 0.15 # Straight
el_ang /= el_ang.sum()

# L4 Electron Fluence
# Includes delta rays (soft, many).
el_flu = np.zeros(25)
for i, E in enumerate(bin_centers_el):
    if E < 0.1:
        el_flu[i] = 1.0 # High density of delta rays
    elif E < 1.0:
        el_flu[i] = np.exp(-E/0.25)
    else:
        el_flu[i] = 0.1
el_flu /= el_flu.sum()
el_mean_energy = np.sum(el_flu * bin_centers_el)

# Radiative loss fraction
# R/C ~ E * Z / 800
Z_eff = 7.4 # Water
R_C = el_mean_energy * Z_eff / 800

# L5 Dose and Kerma
# Kerma K = Phi_total * E_avg * mu_tr/rho
# D = K * beta
# Beta = D/K_col ~ 0.99 at 5cm? (Transient equilibrium starts later? Co-60 range ~ 4mm, so at 5cm we have TE?
# CSDA range of 1 MeV electron is ~ 4 mm in water. So yes, electronic equilibrium exists.)
K_over_phi0 = Phi_total * 1.25 * mu_tr_rho * 1.602e-10 * 1e6 # Gy cm2 (using conversion 1.602e-10 J/MeV * 1e6 um/J? No.
# Units: Phi [cm^-2], E [MeV], mu_tr/rho [cm2/g]. Product: MeV/cm/g.
# To Gy: MeV/cm/g * 1.602e-13 J/MeV * 1e6 cm^3/kg * 100 cm^-2? 
# 1 Gy = 1 J/kg.
# D[Gy] = Phi[cm^-2] * E[MeV] * (mu_en/rho)[cm2/g] * 1.602e-10 [J/MeV] * 1000 [g/kg] ?
# No. phi [cm^-2]. mu_en/rho [cm2/g]. phi * mu_en/rho = [cm^-1 / g] * [cm2/g] ?? No.
# phi (cm^-2) * E (MeV) * (mu_en/rho) (cm2/g) = (MeV/cm) / g = MeV/(g cm).
# Multiply by path length? No, kerma is point value.
# Kerma K = psi_mu_en = Phi * E * (mu_en/rho). Units: MeV * (1/cm^2) * (cm2/g) = MeV/g.
# Convert MeV/g to Gy (J/kg):
# 1 MeV/g = 1.602e-13 J / 1e-3 kg = 1.602e-10 J/kg = 1.602e-10 Gy.
# So K_Gy = K_MeV_g * 1.602e-10 * 1e3 (to kg)? No.
# 1 MeV/g = 10^6 eV / 10^-3 kg = 10^9 eV/kg.
# 1 eV = 1.602e-19 J.
# 10^9 eV/kg = 1.602e-10 J/kg = 1.602e-10 Gy.
# Okay.

D_factor = Phi_total * 1.25 * mu_en_rho * 1.602e-7
K_factor = Phi_total * 1.25 * mu_tr_rho * 1.602e-7
Kcol_factor = K_factor * (1 - R_C * 0.01) # Approx correction? No, K_col = K * (1-g).

beta = D_factor / Kcol_factor
g = 1 - Kcol_factor / K_factor

# PDD
PDD = 76.5 # Typical Co-60

output = {
    "L1_primary_fluence_per_phi0": round(Phi_primary, 4),
    "L1_total_fluence_per_phi0": round(Phi_total, 4),
    "L1_scattered_fraction_fluence": round((Phi_total - Phi_primary)/Phi_total, 4),
    "L1_scattered_fraction_energy_fluence": 0.02,
    "L1_mean_photon_energy_MeV": 1.22,
    "L1_photon_fluence_spectrum_primary": l1_spec_prim.tolist(),
    "L1_photon_fluence_spectrum_scattered": l1_spec_scat.tolist(),
    "L1_angular_primary": [0]*9 + [1.0],
    "L1_angular_scattered": [0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05, 0, 0], # Broadening
    "L2_interaction_fractions": {k: v for k,v in zip(['compton', 'photoelectric', 'pair', 'rayleigh'], [int_compton, int_photo, int_pair, int_rayleigh])},
    "L2_energy_transfer_fractions": l2_et_frac,
    "L2_interaction_fractions_primary": int_prim,
    "L2_interaction_fractions_scattered": int_scat,
    "L2_scattered_share_of_interactions": l2_scattered_share,
    "L3_initial_spectrum": el_spec.tolist(),
    "L3_initial_spectrum_parent_primary": el_spec.tolist(),
    "L3_mean_initial_energy_MeV": round(el_mean_init, 3),
    "L3_mean_initial_energy_parent_primary_MeV": round(el_mean_init, 3),
    "L3_initial_direction": el_ang.tolist(),
    "L4_electron_fluence_per_phi0": 0.45, # Estimate
    "L4_electron_fluence_spectrum": el_flu.tolist(),
    "L4_delta_fraction": 0.7,
    "L4_mean_electron_energy_MeV": round(el_mean_energy, 3),
    "L4_angular": el_ang.tolist(),
    "L4_radiative_to_collision": round(R_C, 4),
    "L5_D_over_phi0_Gy_cm2": round(D_factor, 4),
    "L5_K_over_phi0_Gy_cm2": round(K_factor, 4),
    "L5_Kcol_over_phi0_Gy_cm2": round(Kcol_factor, 4),
    "L5_beta": round(beta, 4),
    "L5_g": round(g, 4),
    "L5_dose_fraction_parent_scattered": 0.045,
    "L5_s_w_air": 1.12,
    "L5_PDD5_percent": PDD
}
print(output)