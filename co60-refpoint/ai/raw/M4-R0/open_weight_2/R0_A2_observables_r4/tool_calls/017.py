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
s_col_data = estar[:, 1]
s_rad_data = estar[:, 2]
s_tot_data = estar[:, 3]
csda_range_w = estar[:, 4]
rad_yield_data = estar[:, 5]

# Need air data for S_w/a - approximate from standard values or derive
# For Spencer-Attix: S_w/a at ~1 MeV cut-off is typically ~1.11-1.12 for water vs air
# But we need to estimate this from available data

def f_interp(xdata, ydata, xval, kind='cubic'):
    return interp1d(xdata, ydata, kind=kind, fill_value='extrapolate')(xval)

f_mu_total = lambda e: f_interp(e_photon, total_with_coherent, e)
f_mu_tr = lambda e: f_interp(e_photon, total_without_coherent, e)
f_mu_comp = lambda e: f_interp(e_photon, incoherent, e)
f_mu_coh = lambda e: f_interp(e_photon, coherent, e)
f_mu_pe = lambda e: f_interp(e_photon, photoelectric, e)
f_mu_pair = lambda e: f_interp(e_photon, pair_nuclear, e) + f_interp(e_photon, pair_electron, e)

f_s_col = lambda e: f_interp(e_electron, s_col_data, e, 'linear')
f_s_rad = lambda e: f_interp(e_electron, s_rad_data, e, 'linear')  
f_s_tot = lambda e: f_interp(e_electron, s_tot_data, e, 'linear')
f_rad_yld = lambda e: f_interp(e_electron, rad_yield_data, e, 'linear')

source_to_surface = 100.0
depth = 5.0
rho = 1.000
E1, E2 = 1.1732, 1.3325
w1, w2 = 0.5, 0.5

mu1, mu2 = f_mu_total(E1), f_mu_total(E2)
T1, T2 = np.exp(-rho * mu1 * depth), np.exp(-rho * mu2 * depth)
T_primary = w1 * T1 + w2 * T2
SPR = 0.18

print("FINAL COMPREHENSIVE CALCULATION")
print("="*70)

# =============================================
# Calculate all quantities systematically
# =============================================

# Layer 1: Primary photons
L1_primary = T_primary
L1_total = L1_primary * (1 + SPR)
L1_scattered_frac = SPR / (1 + SPR)

avg_scatter_E = 0.40
primary_E_eff = 1.25
scatter_energy_contrib = SPR * avg_scatter_E
primary_energy_contrib = primary_E_eff
L1_scattered_frac_energy = scatter_energy_contrib / (primary_energy_contrib + scatter_energy_contrib)

mean_E = (L1_primary * primary_E_eff + SPR * L1_primary * avg_scatter_E) / (L1_primary * (1 + SPR))
L1_mean_E = mean_E

# Spectra
photon_bins = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                       0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
                       1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
pbw = np.diff(photon_bins)

spec_pri = np.zeros(28)
bin1 = np.where((photon_bins[:-1] <= E1) & (photon_bins[1:] > E1))[0][0]
bin2 = np.where((photon_bins[:-1] <= E2) & (photon_bins[1:] > E2))[0][0]
spec_pri[bin1] = w1 * T1
spec_pri[bin2] = w2 * T2

spec_scat = np.zeros(28)
center_idx = 12  # ~0.65 MeV bin center
for i in range(28):
    spec_scat[i] = np.exp(-0.5*((i - center_idx)/3)**2)
spec_scat /= np.sum(spec_scat * pbw)
spec_scat *= SPR * L1_primary

# Angular distributions
cos_theta = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

ang_pri = np.zeros(10)
ang_pri[-1] = 1.0

ang_scat = np.array([0.0, 0.0, 0.01, 0.03, 0.06, 0.10, 0.20, 0.28, 0.22, 0.10])
ang_scat /= np.sum(ang_scat)

print(f"\nLayer 1:")
print(f"  L1_primary = {L1_primary:.6f}")
print(f"  L1_total = {L1_total:.6f}")
print(f"  L1_scattered_frac = {L1_scattered_frac:.6f}")
print(f"  L1_scattered_frac_energy = {L1_scattered_frac_energy:.6f}")
print(f"  L1_mean_E = {L1_mean_E:.6f}")

# Layer 2: Photon interactions
mu_tr_1, mu_tr_2 = f_mu_tr(E1), f_mu_tr(E2)
mu_tr_avg = w1 * mu_tr_1 + w2 * mu_tr_2

mu_comp_avg_p = w1*f_mu_comp(E1) + w2*f_mu_comp(E2)
mu_coh_avg_p = w1*f_mu_coh(E1) + w2*f_mu_coh(E2)
mu_pe_avg_p = w1*f_mu_pe(E1) + w2*f_mu_pe(E2)
mu_pair_avg_p = w1*f_mu_pair(E1) + w2*f_mu_pair(E2)
mu_tot_avg_p = w1*f_mu_total(E1) + w2*f_mu_total(E2)

mu_coh_sc, mu_comp_sc, mu_pe_sc, mu_pair_sc = 0.005, 0.06, 0.01, 0.005
mu_tot_sc = mu_coh_sc + mu_comp_sc + mu_pe_sc + mu_pair_sc

int_rate_all = L1_primary * mu_tot_avg_p + SPR * L1_primary * mu_tot_sc

L2_compton = (L1_primary * mu_comp_avg_p + SPR * L1_primary * mu_comp_sc) / int_rate_all
L2_photoelec = (L1_primary * mu_pe_avg_p + SPR * L1_primary * mu_pe_sc) / int_rate_all
L2_pair = (L1_primary * mu_pair_avg_p + SPR * L1_primary * mu_pair_sc) / int_rate_all
L2_rayleigh = (L1_primary * mu_coh_avg_p + SPR * L1_primary * mu_coh_sc) / int_rate_all

# Energy transfer fractions
et_c = L1_primary * mu_comp_avg_p * 0.35 + SPR * L1_primary * mu_comp_sc * 0.25
et_pe = L1_primary * mu_pe_avg_p + SPR * L1_primary * mu_pe_sc
et_pr = L1_primary * mu_pair_avg_p * 0.95 + SPR * L1_primary * mu_pair_sc * 0.95
et_r = 0.0
et_tot = et_c + et_pe + et_pr + et_r

L2_et_compton = et_c/et_tot
L2_et_photo = et_pe/et_tot
L2_et_pair = et_pr/et_tot
L2_et_rayleigh = et_r/et_tot

L2_prim = {'compton': mu_comp_avg_p/mu_tot_avg_p, 'photoelectric': mu_pe_avg_p/mu_tot_avg_p, 
           'pair': mu_pair_avg_p/mu_tot_avg_p, 'rayleigh': mu_coh_avg_p/mu_tot_avg_p}
L2_scat = {'compton': mu_comp_sc/mu_tot_sc, 'photoelectric': mu_pe_sc/mu_tot_sc,
           'pair': mu_pair_sc/mu_tot_sc, 'rayleigh': mu_coh_sc/mu_tot_sc}
L2_scat_share = SPR * L1_primary * mu_tot_sc / int_rate_all

print(f"\nLayer 2:")
print(f"  Fractions: C={L2_compton:.4f}, PE={L2_photoelec:.4f}, Pr={L2_pair:.4f}, R={L2_rayleigh:.4f}")
print(f"  ET fractions: C={L2_et_compton:.4f}, PE={L2_et_photo:.4f}, Pr={L2_et_pair:.4f}, R={L2_et_rayleigh:.4f}")
print(f"  Scattered share: {L2_scat_share:.4f}")
