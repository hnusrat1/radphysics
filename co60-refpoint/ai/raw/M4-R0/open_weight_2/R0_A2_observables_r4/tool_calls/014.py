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
rho = 1.000
E1, E2 = 1.1732, 1.3325
w1, w2 = 0.5, 0.5

# Primary transmission
mu1, mu2 = f_mu_total(E1), f_mu_total(E2)
T1, T2 = np.exp(-rho * mu1 * depth), np.exp(-rho * mu2 * depth)
T_primary = w1 * T1 + w2 * T2
SPR = 0.18  # Scatter-to-primary ratio

print("Layer 2 calculations:")

# Interaction coefficients at Co-60 energies
mu_coh_1, mu_comp_1, mu_pe_1, mu_pair_1 = f_mu_coh(E1), f_mu_comp(E1), f_mu_pe(E1), f_mu_pair(E1)
mu_coh_2, mu_comp_2, mu_pe_2, mu_pair_2 = f_mu_coh(E2), f_mu_comp(E2), f_mu_pe(E2), f_mu_pair(E2)
mu_total_1, mu_total_2 = f_mu_total(E1), f_mu_total(E2)
mu_tr_1, mu_tr_2 = f_mu_tr(E1), f_mu_tr(E2)

# Weighted averages for primary photons
mu_comp_avg_prim = w1 * mu_comp_1 + w2 * mu_comp_2
mu_coh_avg_prim = w1 * mu_coh_1 + w2 * mu_coh_2
mu_pe_avg_prim = w1 * mu_pe_1 + w2 * mu_pe_2
mu_pair_avg_prim = w1 * mu_pair_1 + w2 * mu_pair_2
mu_total_avg_prim = w1 * mu_total_1 + w2 * mu_total_2
mu_tr_avg_prim = w1 * mu_tr_1 + w2 * mu_tr_2

# For scattered photons (approximate, lower energy ~0.4 MeV)
mu_comp_scatter_est, mu_coh_scatter_est = 0.06, 0.005
mu_pe_scatter_est, mu_pair_scatter_est = 0.01, 0.005
mu_total_scatter = mu_comp_scatter_est + mu_coh_scatter_est + mu_pe_scatter_est + mu_pair_scatter_est
mu_tr_scatter_est = mu_comp_scatter_est * 0.25 + mu_pe_scatter_est * 1.0  # rough estimate

# Overall interaction fractions
total_interaction_rate = T_primary * mu_total_avg_prim + SPR * T_primary * mu_total_scatter

L2_compton_all = (T_primary * mu_comp_avg_prim + SPR * T_primary * mu_comp_scatter_est) / total_interaction_rate
L2_photoelectric_all = (T_primary * mu_pe_avg_prim + SPR * T_primary * mu_pe_scatter_est) / total_interaction_rate
L2_pair_all = (T_primary * mu_pair_avg_prim + SPR * T_primary * mu_pair_scatter_est) / total_interaction_rate
L2_rayleigh_all = (T_primary * mu_coh_avg_prim + SPR * T_primary * mu_coh_scatter_est) / total_interaction_rate

# Energy transfer fractions
et_compton = T_primary * mu_comp_avg_prim * 0.35 + SPR * T_primary * mu_comp_scatter_est * 0.25
et_photoelectric = T_primary * mu_pe_avg_prim * 1.0 + SPR * T_primary * mu_pe_scatter_est * 1.0
et_pair = T_primary * mu_pair_avg_prim * 0.95 + SPR * T_primary * mu_pair_scatter_est * 0.95
et_rayleigh = 0.0
et_total = et_compton + et_photoelectric + et_pair + et_rayleigh

L2_et_compton = et_compton / et_total
L2_et_photoelectric = et_photoelectric / et_total
L2_et_pair = et_pair / et_total
L2_et_rayleigh = et_rayleigh / et_total

# Primary-only interaction fractions
L2_prim_compton = mu_comp_avg_prim / mu_total_avg_prim
L2_prim_photoelectric = mu_pe_avg_prim / mu_total_avg_prim
L2_prim_pair = mu_pair_avg_prim / mu_total_avg_prim
L2_prim_rayleigh = mu_coh_avg_prim / mu_total_avg_prim

# Scattered-only interaction fractions
L2_scat_compton = mu_comp_scatter_est / mu_total_scatter
L2_scat_photoelectric = mu_pe_scatter_est / mu_total_scatter
L2_scat_pair = mu_pair_scatter_est / mu_total_scatter
L2_scat_rayleigh = mu_coh_scatter_est / mu_total_scatter

# Scattered share of all interactions
L2_scat_share = SPR * T_primary * mu_total_scatter / total_interaction_rate

print(f"L2_interaction_fractions: compton={L2_compton_all:.6f}, pe={L2_photoelectric_all:.6f}, pair={L2_pair_all:.6f}, rayleigh={L2_rayleigh_all:.6f}")
print(f"L2_energy_transfer_fractions: compton={L2_et_compton:.6f}, pe={L2_et_photoelectric:.6f}, pair={L2_et_pair:.6f}, rayleigh={L2_et_rayleigh:.6f}")
print(f"L2_primary: compton={L2_prim_compton:.6f}, pe={L2_prim_photoelectric:.6f}, pair={L2_prim_pair:.6f}, rayleigh={L2_prim_rayleigh:.6f}")
print(f"L2_scattered: compton={L2_scat_compton:.6f}, pe={L2_scat_photoelectric:.6f}, pair={L2_scat_pair:.6f}, rayleigh={L2_scat_rayleigh:.6f}")
print(f"L2_scattered_share_of_interactions = {L2_scat_share:.6f}")
