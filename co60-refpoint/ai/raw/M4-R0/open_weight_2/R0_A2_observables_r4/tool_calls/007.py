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

# Create interpolation functions
f_mu_coherent = interp1d(e_photon, coherent, kind='cubic', fill_value='extrapolate')
f_mu_incoherent = interp1d(e_photon, incoherent, kind='cubic', fill_value='extrapolate')
f_mu_photoelectric = interp1d(e_photon, photoelectric, kind='cubic', fill_value='extrapolate')
f_mu_pair_nuc = interp1d(e_photon, pair_nuclear, kind='cubic', fill_value='extrapolate')
f_mu_pair_elt = interp1d(e_photon, pair_electron, kind='cubic', fill_value='extrapolate')
f_mu_total = interp1d(e_photon, total_with_coherent, kind='cubic', fill_value='extrapolate')
f_mu_tr = interp1d(e_photon, total_without_coherent, kind='cubic', fill_value='extrapolate')

f_s_col = interp1d(e_electron, s_col, kind='linear', fill_value='extrapolate')
f_s_rad = interp1d(e_electron, s_rad, kind='linear', fill_value='extrapolate')
f_s_tot = interp1d(e_electron, s_tot, kind='linear', fill_value='extrapolate')

# Geometry parameters
source_to_surface = 100.0  # cm
depth = 5.0  # cm
source_to_point = source_to_surface + depth  # 105 cm
scoring_r = 1.0  # cm radius
scoring_halfz = 0.1  # cm half-thickness (centered at 5 cm)
r_field = 5.0  # cm (field is 10x10 cm, so half-width)

# Density
rho_water = 1.000  # g/cm^3

# Co-60 energies and weights
E1, E2 = 1.1732, 1.3325
w1, w2 = 0.5, 0.5

print("Geometry:")
print(f"  Source-to-surface: {source_to_surface} cm")
print(f"  Depth: {depth} cm")
print(f"  Source-to-point: {source_to_point} cm")
print(f"  Field half-width: {r_field} cm")
print()

# Primary photon fluence calculation
# At distance d from point source, fluence = N / (4*pi*d^2) where N is photons emitted
# Phi0 is defined as primary fluence at 105 cm with phantom removed
# In vacuum, there's no attenuation, so Phi(d=105) = Phi0
# In water, we need to attenuate through 5 cm of water

# Linear attenuation coefficient for Co-60 averaged
mu_linear_avg = rho_water * (w1 * f_mu_total(E1) + w2 * f_mu_total(E2))
print(f"Linear attenuation coefficient (mu): {mu_linear_avg:.4f} cm^-1")

# Primary transmission factor through 5 cm
T_primary = np.exp(-mu_linear_avg * depth)
print(f"Primary transmission through 5 cm water: {T_primary:.6f}")
print(f"L1_primary_fluence_per_phi0 ≈ {T_primary:.6f}")

# Calculate exact primary spectrum
# Primary fluence in each energy bin = weighted sum of attenuated monoenergetic beams
# For monoenergetic: fluence = Phi0_bin * exp(-mu(E)*d)
# Phi0 = Phi0_1 + Phi0_2 = total primary fluence (both lines together)
# So Phi0_1 = w1*Phi0, Phi0_2 = w2*Phi0

mu1_total = f_mu_total(E1)
mu2_total = f_mu_total(E2)
T1 = np.exp(-rho_water * mu1_total * depth)
T2 = np.exp(-rho_water * mu2_total * depth)

print(f"\nEnergy-specific transmissions:")
print(f"  T(1.1732 MeV) = {T1:.6f}")
print(f"  T(1.3325 MeV) = {T2:.6f}")

# Weighted average transmission (for total primary)
T_avg = w1 * T1 + w2 * T2
print(f"  Weighted avg T = {T_avg:.6f}")
print(f"  Confirmed L1_primary_fluence_per_phi0 = {T_avg:.6f}")
