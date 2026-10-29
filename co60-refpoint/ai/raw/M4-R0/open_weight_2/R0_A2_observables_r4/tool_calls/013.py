import numpy as np
from scipy.interpolate import interp1d

# Load data
xcom = np.loadtxt('/tmp/data/xcom_water.txt')
estar = np.loadtxt('/tmp/data/estar_water.txt')

e_photon = xcom[:, 0]
coherent = xcom[:, 1]
incoherent = xcom[:, 2]
photoelectric = xcom[:, 3]
pair_nuclear = xcom[:, 4]
pair_electron = xcom[:, 5]
total_with_coherent = xcom[:, 6]
total_without_coherent = xcom[:, 7]

e_electron = estar[:, 0]
s_col = estar[:, 1]
s_rad = estar[:, 2]
s_tot = estar[:, 3]
csda_range = estar[:, 4]
rad_yield_data = estar[:, 5]

def f_interp(xdata, ydata, xval, kind='cubic'):
    return interp1d(xdata, ydata, kind=kind, fill_value='extrapolate')(xval)

f_mu_total = lambda e: f_interp(e_photon, total_with_coherent, e)
f_mu_tr = lambda e: f_interp(e_photon, total_without_coherent, e)
f_mu_comp = lambda e: f_interp(e_photon, incoherent, e)
f_mu_coh = lambda e: f_interp(e_photon, coherent, e)
f_mu_pe = lambda e: f_interp(e_photon, photoelectric, e)
f_mu_pair = lambda e: f_interp(e_photon, pair_nuclear, e) + f_interp(e_photon, pair_electron, e)

f_s_col = lambda e: f_interp(e_electron, s_col, e, 'linear')
f_s_rad = lambda e: f_interp(e_electron, s_rad, e, 'linear')  
f_s_tot = lambda e: f_interp(e_electron, s_tot, e, 'linear')
f_rad_yld = lambda e: f_interp(e_electron, rad_yield_data, e, 'linear')

source_to_surface = 100.0
depth = 5.0
source_to_point = 105.0
rho = 1.000
E1, E2 = 1.1732, 1.3325
w1, w2 = 0.5, 0.5

print("\n" + "="*80)
print("LAYER 2: PHOTON INTERACTIONS AT THE POINT")
print("="*80)

# Interaction coefficients at Co-60 energies
mu_coh_1, mu_comp_1, mu_pe_1, mu_pair_1 = f_mu_coh(E1), f_mu_comp(E1), f_mu_pe(E1), f_mu_pair(E1)
mu_coh_2, mu_comp_2, mu_pe_2, mu_pair_2 = f_mu_coh(E2), f_mu_comp(E2), f_mu_pe(E2), f_mu_pair(E2)
mu_total_1, mu_total_2 = f_mu_total(E1), f_mu_total(E2)

# Weighted interaction fractions for ALL photons at point (primary + scattered)
# Primary: dominated by Compton (~99.9%), minor pair production
# Scattered: lower energy, more Compton relative contribution
# Approximate weighted averages
mu_comp_avg_prim = w1 * mu_comp_1 + w2 * mu_comp_2
mu_coh_avg_prim = w1 * mu_coh_1 + w2 * mu_coh_2
mu_pe_avg_prim = w1 * mu_pe_1 + w2 * mu_pe_2
mu_pair_avg_prim = w1 * mu_pair_1 + w2 * mu_pair_2
mu_total_avg_prim = w1 * mu_total_1 + w2 * mu_total_2

print(f"\nPRIMARY PHOTON INTERACTION FRACTIONS:")
print(f"  Compton: {mu_comp_avg_prim/mu_total_avg_prim:.6f}")
print(f"  Photoelectric: {mu_pe_avg_prim/mu_total_avg_prim:.6f}")
print(f"  Pair: {(mu_pair_avg_prim)/mu_total_avg_prim:.6f}")
print(f"  Rayleigh: {mu_coh_avg_prim/mu_total_avg_prim:.6f}")

# For ALL photons (including scatter), Compton fraction is even higher at lower energies
mu_comp_scatter_est = 0.98  # approximate for ~0.4 MeV photons
mu_coh_scatter_est = 0.005  # increases at lower energies
mu_pe_scatter_est = 0.01   # becomes significant below 0.5 MeV
mu_pair_scatter_est = 0.005
mu_total_scatter = mu_comp_scatter_est + mu_coh_scatter_est + mu_pe_scatter_est + mu_pair_scatter_est

# Overall fractions (weighted average of primary and scatter interactions)
frac_all_compton = (T_primary * mu_comp_avg_prim + SPR * T_primary * mu_comp_scatter_est) / \
                   (T_primary * mu_total_avg_prim + SPR * T_primary * mu_total_scatter)
frac_all_photoelectric = (T_primary * mu_pe_avg_prim + SPR * T_primary * mu_pe_scatter_est) / \
                         (T_primary * mu_total_avg_prim + SPR * T_primary * mu_total_scatter)
frac_all_pair = (T_primary * mu_pair_avg_prim + SPR * T_primary * mu_pair_scatter_est) / \
                (T_primary * mu_total_avg_prim + SPR * T_primary * mu_total_scatter)
frac_all_rayleigh = (T_primary * mu_coh_avg_prim + SPR * T_primary * mu_coh_scatter_est) / \
                    (T_primary * mu_total_avg_prim + SPR * T_primary * mu_total_scatter)

print(f"\nALL PHOTONS INTERACTION FRACTIONS:")
print(f"  Compton: {frac_all_compton:.6f}")
print(f"  Photoelectric: {frac_all_photoelectric:.6f}")
print(f"  Pair: {frac_all_pair:.6f}")
print(f"  Rayleigh: {frac_all_rayleigh:.6f}")

# Energy transfer fractions - need mass energy-transfer coefficients
mu_tr_1, mu_tr_2 = f_mu_tr(E1), f_mu_tr(E2)
mu_tr_avg_prim = w1 * mu_tr_1 + w2 * mu_tr_2

# For scattering: rough estimates for energy transfer at lower energies
mu_tr_comp_scatter = mu_comp_scatter_est * 0.6  # ~60% energy transfer per Compton scatter
mu_tr_coh_scatter = 0  # no energy transfer in coherent scattering
mu_tr_pe_scatter = mu_pe_scatter_est * 1.0  # full transfer in PE
mu_tr_pair_scatter = mu_pair_scatter_est * 0.95  # E-2m_ec^2 transferred ≈ 95%

mu_tr_total_scatter = mu_tr_comp_scatter + mu_tr_coh_scatter + mu_tr_pe_scatter + mu_tr_pair_scatter

mu_tr_all = T_primary * mu_tr_avg_prim + SPR * T_primary * mu_tr_total_scatter

# Fractions weighted by energy transfer
et_frac_compton = T_primary * mu_tr_1 * 0.3 + SPR * T_primary * mu_tr_comp_scatter
# Wait, need better approach using actual mu_tr values per process

# Let's use a simpler approach: 
# Energy transfer for each process = (mu_tr_process / mu_tr_total) * mu_tr_total
# where mu_tr_process accounts for partial transfer in Compton

# For simplicity, assume similar proportional distribution but scaled by process-specific transfer fractions
# Compton at ~1 MeV transfers ~30-40% of photon energy on average
# Compton at ~0.4 MeV transfers ~20-30%
# Photoelectric: 100%
# Pair: ~95% (E - 2m_e c^2)

transfer_fractions_per_interaction = {
    'compton': 0.35,  # avg fraction of E transferred in Compton
    'photoelectric': 1.0,
    'pair': 0.95,
    'rayleigh': 0.0
}

et_num_compton = T_primary * mu_comp_avg_prim * transfer_fractions_per_interaction['compton'] + \
                 SPR * T_primary * mu_comp_scatter_est * 0.25
et_num_photoelectric = T_primary * mu_pe_avg_prim * 1.0 + SPR * T_primary * mu_pe_scatter_est * 1.0
et_num_pair = T_primary * mu_pair_avg_prim * 0.95 + SPR * T_primary * mu_pair_scatter_est * 0.95
et_num_rayleigh = 0  # no energy transfer

et_total = et_num_compton + et_num_photoelectric + et_num_pair + et_num_rayleigh

et_frac_compton = et_num_compton / et_total
et_frac_photoelectric = et_num_photoelectric / et_total
et_frac_pair = et_num_pair / et_total
et_frac_rayleigh = et_num_rayleigh / et_total

print(f"\nENERGY TRANSFER FRACTIONS:")
print(f"  Compton: {et_frac_compton:.6f}")
print(f"  Photoelectric: {et_frac_photoelectric:.6f}")
print(f"  Pair: {et_frac_pair:.6f}")
print(f"  Rayleigh: {et_frac_rayleigh:.6f}")

# Interactions by process for primary only (already computed above)
L2_interaction_fractions_primary = {
    'compton': mu_comp_avg_prim / mu_total_avg_prim,
    'photoelectric': mu_pe_avg_prim / mu_total_avg_prim,
    'pair': mu_pair_avg_prim / mu_total_avg_prim,
    'rayleigh': mu_coh_avg_prim / mu_total_avg_prim
}

# For scattered photons only
L2_interaction_fractions_scattered = {
    'compton': mu_comp_scatter_est / mu_total_scatter,
    'photoelectric': mu_pe_scatter_est / mu_total_scatter,
    'pair': mu_pair_scatter_est / mu_total_scatter,
    'rayleigh': mu_coh_scatter_est / mu_total_scatter
}

# All photons (overall)
L2_interaction_fractions = {
    'compton': frac_all_compton,
    'photoelectric': frac_all_photoelectric,
    'pair': frac_all_pair,
    'rayleigh': frac_all_rayleigh
}

L2_energy_transfer_fractions = {
    'compton': et_frac_compton,
    'photoelectric': et_frac_photoelectric,
    'pair': et_frac_pair,
    'rayleigh': et_frac_rayleigh
}

# Scattered share of all interactions
all_interactions = T_primary * mu_total_avg_prim + SPR * T_primary * mu_total_scatter
scatter_interactions = SPR * T_primary * mu_total_scatter
L2_scattered_share_of_interactions = scatter_interactions / all_interactions

print(f"\nL2_scattered_share_of_interactions = {L2_scattered_share_of_interactions:.6f}")
