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
scoring_r = 1.0
scoring_halfz = 0.1
rho = 1.000

E1, E2 = 1.1732, 1.3325
w1, w2 = 0.5, 0.5

print("="*70)
print("DETAILED CALCULATIONS - CORRECTED")
print("="*70)

# LAYER 1: PRIMARY PHOTONS
mu1 = f_mu_total(E1)
mu2 = f_mu_total(E2)
T1 = np.exp(-rho * mu1 * depth)
T2 = np.exp(-rho * mu2 * depth)
T_primary = w1 * T1 + w2 * T2

print(f"\nPRIMARY FLUENCE:")
print(f"  T({E1}) = {T1:.6f}")
print(f"  T({E2}) = {T2:.6f}")
print(f"  L1_primary_fluence_per_phi0 = {T_primary:.6f}")

# SCATTERED PHOTON ESTIMATE
# For Co-60 at 5 cm depth in 10x10 cm field:
# Clinical reference data shows PDD(5cm) ≈ 67% for 10x10
# This means dose rate at 5 cm is 67% of dose rate at d_max
# Since d_max ≈ 0.5 cm for Co-60, let's estimate scatter contribution

# Using simplified approach: assume SPR (scatter-to-primary ratio) ≈ 0.18
SPR = 0.18

# Scattered fraction of fluence
scattered_fraction_fluence = SPR / (1 + SPR)

print(f"\nSCATTERED PHOTON APPROXIMATION:")
print(f"  Assumed SPR = {SPR}")
print(f"  L1_total_fluence_per_phi0 = {T_primary*(1+SPR):.6f}")
print(f"  L1_scattered_fraction_fluence = {scattered_fraction_fluence:.6f}")

# Energy spectrum of scattered photons - rough estimate
# Compton scattered photons from ~1 MeV typically have energies 0.2-0.8 MeV
# Weighted average ≈ 0.4 MeV
avg_scatter_E = 0.40

# Energy-weighted scattered fraction
primary_E_eff = 1.25  # weighted mean primary energy
scatter_energy_flux = SPR * T_primary * avg_scatter_E
primary_energy_flux = T_primary * primary_E_eff
total_energy_flux = primary_energy_flux + scatter_energy_flux
scattered_fraction_energy = scatter_energy_flux / total_energy_flux

print(f"  Mean scattered photon energy ≈ {avg_scatter_E} MeV")
print(f"  L1_scattered_fraction_energy_fluence = {scattered_fraction_energy:.6f}")

# Mean photon energy (fluence-weighted)
mean_photon_E = (T_primary * primary_E_eff + SPR * T_primary * avg_scatter_E) / \
                (T_primary * (1 + SPR))
print(f"  L1_mean_photon_energy_MeV = {mean_photon_E:.6f}")

# Photon fluence spectra
photon_energy_bins = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                               0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
                               1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])

spec_primary = np.zeros(28)
bin_1 = np.where((photon_energy_bins[:-1] <= E1) & (photon_energy_bins[1:] > E1))[0][0]
bin_2 = np.where((photon_energy_bins[:-1] <= E2) & (photon_energy_bins[1:] > E2))[0][0]
spec_primary[bin_1] = w1 * T1
spec_primary[bin_2] = w2 * T2

# Scatter spectrum: approximate using broad distribution centered around 0.4 MeV
# Bin width = 0.05 MeV for most bins
spec_scattered = np.zeros(28)
scatter_bin_widths = np.diff(photon_energy_bins)
# Gaussian-like distribution centered at 0.4 MeV with sigma ≈ 0.15 MeV
center_idx = np.argmin(np.abs(photon_energy_bins - 0.4))
sigma = 0.15 / 0.05  # in bin units
for i, e_center in enumerate((photon_energy_bins[:-1] + photon_energy_bins[1:]) / 2):
    spec_scattered[i] = np.exp(-0.5 * ((i - center_idx) / sigma)**2)
# Normalize so integral = SPR * T_primary
integral = np.sum(spec_scattered * scatter_bin_widths)
spec_scattered *= (SPR * T_primary) / integral

print(f"\nSPECTRA SUMS:")
print(f"  Primary spectrum sum = {np.sum(spec_primary * scatter_bin_widths):.6f}")
print(f"  Scattered spectrum sum = {np.sum(spec_scattered * scatter_bin_widths):.6f}")
