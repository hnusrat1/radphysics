import numpy as np

# NIST XCOM data for Water (H2O) at 2.0 MeV
# Values from nist.gov/xcom (approximate standard values)
rho = 1.0 # g/cm3
d = 10.0 # cm
# Mass attenuation coefficients (cm^2/g)
mu_rhoy_total = 0.06366
mu_rhy_rayleigh = 0.00213
mu_rhy_compton = 0.05876
mu_rhy_photo = 0.00011
# Energy absorption/tranfer coefficients
mu_en_rho = 0.02966
mu_tr_rho = 0.02981

mu = mu_rhoy_total * rho
mu_en = mu_en_rho * rho
mu_tr = mu_tr_rho * rho

# 1. Primary fluence
primary_ratio = np.exp(-mu * d)
print(f"L1_primary_fluence_per_phi0: {primary_ratio:.4f}")

# 2. Buildup factor for fluence at 10 cm, 2 MeV, water
# Using Berger's parameterization or Moyer: typically ~1.65 for 2 MeV at 10 cm
B_f = 1.65
total_fluence_ratio = primary_ratio * B_f
scat_frac_fluence = 1.0 - 1.0/B_f
print(f"L1_total: {total_fluence_ratio:.4f}, scat_frac_fluence: {scat_frac_fluence:.4f}")

# Energy fluence ratio: scattered photons are softer
# Assume avg E_scat ~ 0.8 MeV, avg E_prim = 2.0 MeV
avg_E_prim = 2.0
avg_E_scat = 0.85
energy_prim_contrib = primary_ratio * avg_E_prim
energy_scat_contrib = (total_fluence_ratio - primary_ratio) * avg_E_scat
total_energy = energy_prim_contrib + energy_scat_contrib
scat_frac_energy = energy_scat_contrib / total_energy
print(f"L1_scat_frac_energy: {scat_frac_energy:.4f}")

# Mean photon energy
mean_E = total_energy / total_fluence_ratio
print(f"L1_mean_E: {mean_E:.4f}")

# 3. Interaction fractions (normalized to sum to 1)
sig_compton = mu_rhy_compton
sig_rayleigh = mu_rhy_rayleigh
sig_photo = mu_rhy_photo
sig_pair = 0.0
sig_sum = sig_compton + sig_rayleigh + sig_photo
F_compton = sig_compton/sig_sum
F_rayleigh = sig_rayleigh/sig_sum
F_photo = sig_photo/sig_sum
F_pair = 0.0
print(f"Interactions: compton {F_compton:.3f}, rayleigh {F_rayleigh:.3f}, photo {F_photo:.3f}")

# Energy transfer fractions (weighted by E_trans)
# Rayleigh transfers 0. PE transfers ~Z binding + KE ~ 100% of E_gamma for low E, but at 2MeV PE is negligible.
# Compton E_trans fraction ~ mu_tr/mu_compton_effective ~ 0.78
w_compton = F_compton * 0.78
w_pe = F_photo * 1.0
w_ray = 0.0
w_sum = w_compton + w_pe
print(f"E-transfer: compton {w_compton/w_sum:.3f}, photo {w_pe/w_sum:.3f}")

# 4. Kerma & Dose
K_col = total_fluence_ratio * primary_ratio * mu_en # wait, K_col uses local fluence. 
# Local total fluence is phi_total = primary * B_f.
# K_col = phi_total * (mu_en/rho)*rho = phi_total * mu_en_rho
K_col_val = total_fluence_ratio * mu_en_rho * rho # per Phi0
print(f"L5_Kcol_over_phi0: {K_col_val:.4f} Gy cm^2")

# D = K_col * beta. beta ~ 0.95 at 10 cm for 2 MeV.
beta = 0.96
D_val = K_col_val * beta
print(f"L5_D_over_phi0: {D_val:.4f}")

g = 1 - beta # approximate
print(f"L5_g: {g:.4f}")
