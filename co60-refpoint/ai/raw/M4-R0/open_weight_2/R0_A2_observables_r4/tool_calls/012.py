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

# Geometry & constants
source_to_surface = 100.0
depth = 5.0
source_to_point = 105.0
rho = 1.000
E1, E2 = 1.1732, 1.3325
w1, w2 = 0.5, 0.5

print("="*80)
print("ALL LAYER CALCULATIONS")
print("="*80)

# ==========================================
# LAYER 1: Photons at the point
# ==========================================
mu1 = f_mu_total(E1)
mu2 = f_mu_total(E2)
T1 = np.exp(-rho * mu1 * depth)
T2 = np.exp(-rho * mu2 * depth)
T_primary = w1 * T1 + w2 * T2

SPR = 0.18  # Scatter-to-primary ratio (estimate for Co-60, 10x10 field, 5 cm)

# Primary fluence
L1_primary_fluence_per_phi0 = T_primary

# Total fluence = primary + scattered
L1_total_fluence_per_phi0 = T_primary * (1 + SPR)

# Scattered fraction (fluence)
L1_scattered_fraction_fluence = SPR / (1 + SPR)

# Scattered fraction of energy fluence
avg_scatter_E = 0.40  # approximate mean scattered photon energy
primary_E_eff = 1.25  # weighted average primary energy
scatter_energy_contrib = SPR * avg_scatter_E
primary_energy_contrib = primary_E_eff
L1_scattered_fraction_energy_fluence = scatter_energy_contrib / (primary_energy_contrib + scatter_energy_contrib)

# Mean photon energy
mean_E_num = T_primary * primary_E_eff + SPR * T_primary * avg_scatter_E
mean_E_den = T_primary * (1 + SPR)
L1_mean_photon_energy_MeV = mean_E_num / mean_E_den

# Spectra
photon_energy_bins = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                               0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
                               1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
scatter_bin_widths = np.diff(photon_energy_bins)

spec_primary = np.zeros(28)
bin_1 = np.where((photon_energy_bins[:-1] <= E1) & (photon_energy_bins[1:] > E1))[0][0]
bin_2 = np.where((photon_energy_bins[:-1] <= E2) & (photon_energy_bins[1:] > E2))[0][0]
spec_primary[bin_1] = w1 * T1
spec_primary[bin_2] = w2 * T2

# Scatter spectrum approximation
spec_scattered = np.zeros(28)
center_idx = int(np.argmin(np.abs((photon_energy_bins[:-1] + photon_energy_bins[1:]) / 2 - 0.4)))
sigma = 3.0  # in bin units (~0.15 MeV)
for i in range(28):
    spec_scattered[i] = np.exp(-0.5 * ((i - center_idx) / sigma)**2)
integral = np.sum(spec_scattered * scatter_bin_widths)
spec_scattered *= (SPR * T_primary) / integral

# Angular distributions
cos_theta_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

# Primary: all along beam axis
angular_primary = np.zeros(10)
angular_primary[-1] = 1.0  # [0.8, 1] bin

# Scatter: broader distribution (approximate)
angular_scattered_approx = np.array([0.0, 0.0, 0.01, 0.03, 0.06, 0.10, 0.20, 0.28, 0.22, 0.10])
angular_scattered = angular_scattered_approx / np.sum(angular_scattered_approx)

print("\n--- LAYER 1 RESULTS ---")
print(f"L1_primary_fluence_per_phi0 = {L1_primary_fluence_per_phi0:.6f}")
print(f"L1_total_fluence_per_phi0 = {L1_total_fluence_per_phi0:.6f}")
print(f"L1_scattered_fraction_fluence = {L1_scattered_fraction_fluence:.6f}")
print(f"L1_scattered_fraction_energy_fluence = {L1_scattered_fraction_energy_fluence:.6f}")
print(f"L1_mean_photon_energy_MeV = {L1_mean_photon_energy_MeV:.6f}")
print(f"Primary spectrum sum = {np.sum(spec_primary * scatter_bin_widths):.6f}")
print(f"Scatter spectrum sum = {np.sum(spec_scattered * scatter_bin_widths):.6f}")
