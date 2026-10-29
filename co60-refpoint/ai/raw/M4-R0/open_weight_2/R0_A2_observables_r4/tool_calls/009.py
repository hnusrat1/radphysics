import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

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

# Interpolation functions
def f_interp(xdata, ydata, xval, kind='cubic'):
    if kind == 'linear':
        return interp1d(xdata, ydata, kind='linear', fill_value='extrapolate')(xval)
    else:
        return interp1d(xdata, ydata, kind='cubic', fill_value='extrapolate')(xval)

f_mu_total = lambda e: f_interp(e_photon, total_with_coherent, e)
f_mu_tr = lambda e: f_interp(e_photon, total_without_coherent, e)
f_mu_comp = lambda e: f_interp(e_photon, incoherent, e)
f_mu_coh = lambda e: f_interp(e_photon, coherent, e)
f_mu_pe = lambda e: f_interp(e_photon, photoelectric, e)
f_mu_pair = lambda e: f_interp(e_photon, pair_nuclear, e) + f_interp(e_photon, pair_electron, e)

f_s_col = lambda e: f_interp(e_electron, s_col, e, 'linear')
f_s_rad = lambda e: f_interp(e_electron, s_rad, e, 'linear')
f_s_tot = lambda e: f_interp(e_electron, s_tot, e, 'linear')

# Geometry
source_to_surface = 100.0  # cm
depth = 5.0  # cm  
source_to_point = 105.0  # cm
scoring_r = 1.0  # cm radius
scoring_halfz = 0.1  # cm half-thickness
rho = 1.000  # g/cm^3

E1, E2 = 1.1732, 1.3325  # MeV
w1, w2 = 0.5, 0.5  # Equal probability

print("="*60)
print("LAYER 1: PHOTONS ARRIVING AT THE POINT")
print("="*60)

# Primary fluence
mu1 = f_mu_total(E1)
mu2 = f_mu_total(E2)
T1 = np.exp(-rho * mu1 * depth)
T2 = np.exp(-rho * mu2 * depth)
T_primary = w1 * T1 + w2 * T2

print(f"\nPrimary transmission:")
print(f"  T({E1}) = {T1:.6f}")
print(f"  T({E2}) = {T2:.6f}")
print(f"  Weighted T = {T_primary:.6f}")
print(f"  L1_primary_fluence_per_phi0 = {T_primary:.6f}")

# For scattered photons, we need an approximation
# In a narrow field (10x10 cm), at 5 cm depth, scatter contribution exists but is modest
# Using typical clinical data approximations for Co-60:
# - Scatter-to-primary ratio (SPR) at 5 cm for 10x10 field ≈ 0.15-0.25
# This depends strongly on exact geometry; let's estimate SPR ≈ 0.20

# For the scoring volume at 5 cm, the scattered fraction would be:
# Fluence_scattered / Fluence_total ≈ SPR / (1 + SPR)

spr_estimate = 0.20  # Approximate scatter-to-primary ratio

# Scattered fraction of fluence
frac_scattered_fluence = spr_estimate / (1 + spr_estimate)
frac_primary_fluence = 1 / (1 + spr_estimate)

print(f"\nScattered photon estimates (approximate):")
print(f"  Assumed SPR = {spr_estimate}")
print(f"  Scattered fraction (fluence) = {frac_scattered_fluence:.4f}")
print(f"  Primary fraction (fluence) = {frac_primary_fluence:.4f}")
print(f"  L1_scattered_fraction_fluence ≈ {frac_scattered_fluence:.4f}")
print(f"  L1_total_fluence_per_phi0 = {T_primary + spr_estimate*T_primary:.6f}")

# Energy-weighted scatter fraction
# Scattered photons are lower energy on average (~0.3-0.8 MeV vs 1.25 MeV primary)
avg_scatter_energy = 0.5  # MeV, rough estimate
energy_scatter_weight = avg_scatter_energy / 1.25  # relative to primary
frac_scattered_energy = (spr_estimate * frac_scattered_fluence * energy_scatter_weight) / \
                        (frac_primary_fluence + spr_estimate * frac_scattered_fluence * energy_scatter_weight)

print(f"  Estimated scattered energy fraction ≈ {frac_scattered_energy:.4f}")

# Mean photon energy
mean_E = (T_primary * 1.25 + spr_estimate * T_primary * 0.5 * avg_scatter_energy/1.25) / \
         (T_primary * (1 + spr_estimate))
# More accurate calculation using actual spectra below

# Angular distributions
# Primary: very narrow cone (solid angle subtended by source point viewed from scoring volume is tiny)
# Actually, Phi0 is defined at the scoring location WITHOUT phantom, so it represents the fluence
# that would be there from the point source. With the phantom present, primary photons arrive
# only along the line from source through the scoring volume.
# Since it's a point source, primary direction is essentially parallel to beam axis
# Within the scoring cylinder r=1cm at z=5cm (5.1 to 4.9 cm depth), the angle spread is small

# Calculate max angle for primary from edge of scoring volume
max_angle_deg = np.arctan(scoring_r / source_to_point) * 180 / np.pi
print(f"\nAngular spread of primaries: ±{max_angle_deg:.2f} degrees")
print(f"  cos(theta_min) = {np.cos(np.radians(max_angle_deg)):.6f}")

# Primary angular distribution (essentially collimated along beam axis)
cos_theta_bins = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
angular_primary = np.zeros(10)
# All primary photons have cos(theta) near 1 (forward direction)
# With our grid, most fall in bin [0.8, 1]
cos_max = np.cos(np.radians(max_angle_deg))
bin_idx_8_1 = np.searchsorted(cos_theta_bins[1:], 1.0)
bin_idx_cosmax = np.searchsorted(cos_theta_bins[1:], cos_max)

# Primary spectrum bins
photon_energy_bins = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                               0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
                               1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])

primary_spectrum = np.zeros(28)
# Bin edges for Co-60 lines
# 1.1732 falls in bin index 23 (1.15-1.2) or needs precise indexing
bin_E1 = np.where((photon_energy_bins[:-1] <= E1) & (photon_energy_bins[1:] > E1))[0][0]
bin_E2 = np.where((photon_energy_bins[:-1] <= E2) & (photon_energy_bins[1:] > E2))[0][0]
print(f"\nSpectrum bin indices: E1={bin_E1} [{photon_energy_bins[bin_E1]:.2f}-{photon_energy_bins[bin_E1+1]:.2f}], E2={bin_E2}")

primary_spectrum[bin_E1] = w1 * T1
primary_spectrum[bin_E2] = w2 * T2
print(f"Primary spectrum (summed per line): {primary_spectrum.sum():.6f}")
