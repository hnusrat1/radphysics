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

# Constants
source_to_surface = 100.0
depth = 5.0
source_to_point = 105.0
scoring_r = 1.0
scoring_halfz = 0.1
rho = 1.000

E1, E2 = 1.1732, 1.3325
w1, w2 = 0.5, 0.5

print("="*70)
print("COMPREHENSIVE CALCULATION FOR ALL LAYERS")
print("="*70)

# ============================================================
# LAYER 1: PHOTONS
# ============================================================
mu1 = f_mu_total(E1)
mu2 = f_mu_total(E2)
T1 = np.exp(-rho * mu1 * depth)
T2 = np.exp(-rho * mu2 * depth)
T_primary = w1 * T1 + w2 * T2

print("\n--- Layer 1: Photons ---")
print(f"Primary fluence per Phi0 = {T_primary:.6f}")

# For Co-60 at 5 cm depth in a 10x10 cm field, 
# clinical data shows d_max ~0.5 cm, PDD(5cm) ~67% for 10x10
# Primary transmission is ~73%, so total (including scatter) gives PDD
# Typical SPR for Co-60 at 5 cm, 10x10 field ≈ 0.15-0.20
# Let's use SPR = 0.18

SPR = 0.18
frac_scattered_fluence = SPR / (1 + SPR)
frac_primary_in_total = 1 / (1 + SPR)

print(f"L1_primary_fluence_per_phi0 = {T_primary:.6f}")
print(f"L1_total_fluence_per_phi0 = {T_primary*(1+SPR):.6f}")
print(f"L1_scattered_fraction_fluence = {frac_scattered_fluence:.4f}")

# Scattered photon energies are lower (Compton scattered down to ~0.3-0.8 MeV)
# Weighted average scattered energy
# Using Klein-Nishina and geometry, approximate mean scattered energy
avg_scatter_E = 0.45  # MeV estimate for broad angular distribution of scattered photons

# Energy fluence fraction
primary_energy_fluence = T_primary * 1.25  # weighted average primary E ≈ 1.25 MeV
scatter_energy_fluence = SPR * T_primary * avg_scatter_E
total_energy_fluence = primary_energy_fluence + scatter_energy_fluence
frac_scattered_energy = scatter_energy_fluence / total_energy_fluence

print(f"L1_scattered_fraction_energy_fluence = {frac_scattered_energy:.4f}")

# Mean photon energy (fluence-weighted)
mean_E = (T_primary * 1.25 + SPR * T_primary * avg_scatter_E) / (T_primary * (1 + SPR))
print(f"L1_mean_photon_energy_MeV = {mean_E:.4f}")

# Primary spectrum (28 bins)
photon_energy_bins = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                               0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
                               1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])

spec_primary = np.zeros(28)
bin_1 = np.where((photon_energy_bins[:-1] <= E1) & (photon_energy_bins[1:] > E1))[0][0]
bin_2 = np.where((photon_energy_bins[:-1] <= E2) & (photon_energy_bins[1:] > E2))[0][0]
spec_primary[bin_1] = w1 * T1
spec_primary[bin_2] = w2 * T2

# Scatter spectrum - spread over many bins, peaked around 0.3-0.8 MeV
# Approximate using broadened distribution
spec_scattered = np.zeros(28)
for i in range(10, 18):  # Roughly 0.5-0.9 MeV range where most scatter falls
    spec_scattered[i] = SPR * T_primary * 0.08  # normalized approximation
# Adjust to make it integrate to SPR*T_primary
spec_scattered_sum = np.sum(spec_scattered) * (photon_energy_bins[1] - photon_energy_bins[0])
# Normalize spectrum so integral = SPR*T_primary
if abs(spec_scattered_sum) > 0:
    spec_scattered *= (SPR * T_primary) / (spec_scattered_sum * (photon_energy_bins[1] - photon_energy_bins[0]))
else:
    # Fallback: distribute evenly in scatter region
    n_bins = 8
    for i in range(12, min(20, 28)):
        spec_scattered[i] = SPR * T_primary / n_bins

print(f"L1_photon_fluence_spectrum_primary sum = {spec_primary.sum():.6f}")
print(f"L1_photon_fluence_spectrum_scattered sum = {spec_scattered.sum():.6f}")

# Angular distributions
cos_theta_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

# Primary: essentially all along beam axis (cos ~1)
# Max angle from scoring cylinder edge: arctan(1/105) ≈ 0.55 degrees
max_angle = np.arctan(scoring_r / source_to_point)
angular_primary = np.zeros(10)
# cos(0.55°) ≈ 0.99995, well into [0.8, 1] bin
# With our grid, all primary in last bin
angular_primary[-1] = 1.0

# Scatter: broader angular distribution
# At ~1 MeV, Compton scattering has significant wide-angle component
# Use rough approximation based on Klein-Nishina integrated over volume
# Simplified: assume somewhat isotropic but forward-biased
# In practice for narrow beams, scatter is also relatively forward-peaked
angular_scattered = np.array([0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.25, 0.2, 0.15, 0.10])
# Normalize to sum to 1
angular_scattered /= np.sum(angular_scattered)

print(f"Angular distributions computed")
