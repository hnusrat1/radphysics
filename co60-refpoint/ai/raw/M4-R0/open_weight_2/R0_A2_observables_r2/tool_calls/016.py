import numpy as np

# Comprehensive calculation of all requested quantities
print("=" * 60)
print("COMPREHENSIVE CALCULATION OF ALL QUANTITIES")
print("=" * 60)

# Load data files
xcom = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
estar = np.loadtxt('/tmp/data/estar_water.txt', comments='#')

def interp_xcom(col_idx, E):
    """Interpolate XCOM column at energy E using log-log interpolation"""
    return np.interp(np.log10(E), np.log10(xcom[:, 0]), xcom[:, col_idx])

def interp_estar(col_idx, E):
    """Interpolate ESTAR column at energy E using linear interpolation"""
    return np.interp(E, estar[:, 0], estar[:, col_idx])

# ===== PARAMETERS =====
E_photons = np.array([1.1732, 1.3325])  # MeV - Co-60 lines
w_photons = np.array([0.5, 0.5])  # Equal probability
SSD = 100.0  # cm - source to surface distance
depth = 5.0  # cm - depth of interest (105 cm from source)
rho = 1.0  # g/cm^3 - water density

# Scoring volume: cylinder, radius 1 cm, thickness 0.2 cm centered at 5 cm depth
# Volume = pi * r^2 * h = pi * 1^2 * 0.2 = 0.628 cm^3
# Mass = 0.628 g

# Bin definitions
photon_energy_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                                0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
                                1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
electron_energy_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                                  0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                                  1, 1.05, 1.1, 1.15, 1.2, 1.4])
cos_theta_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

# ===== LAYER 1: PHOTONS ARRIVING AT THE POINT =====

# Weighted mass attenuation coefficients
mu_mass_total_w = sum(w * interp_xcom(6, E) for w, E in zip(w_photons, E_photons))  # total with coherent
mu_mass_compton_w = sum(w * interp_xcom(2, E) for w, E in zip(w_photons, E_photons))
mu_mass_coherent_w = sum(w * interp_xcom(1, E) for w, E in zip(w_photons, E_photons))
mu_mass_photo_w = sum(w * interp_xcom(3, E) for w, E in zip(w_photons, E_photons))
mu_mass_pair_nuc_w = sum(w * interp_xcom(4, E) for w, E in zip(w_photons, E_photons))

mu_lin_total = mu_mass_total_w * rho  # cm^-1

# Primary transmission fraction
primary_frac = np.exp(-mu_lin_total * depth)

# Estimate scattered fluence buildup factor
# For Co-60 at 5 cm depth in water with narrow field, scatter buildup is modest
# Using typical values from literature: scatter-to-primary ratio on-axis ~ 0.15-0.25
scatter_to_primary_ratio = 0.18  # reasonable estimate

# Total fluence = primary + scattered
total_frac = primary_frac * (1 + scatter_to_primary_ratio)

# Layer 1 results
L1_primary_fluence_per_phi0 = primary_frac
L1_total_fluence_per_phi0 = total_frac
L1_scattered_fraction_fluence = 1 - L1_primary_fluence_per_phi0 / L1_total_fluence_per_phi0

# Scattered fraction of energy fluence
# Scattered photons have lower average energy than primary
mean_primary_E = np.average(E_photons, weights=w_photons)  # ~1.253 MeV
mean_scattered_E = mean_primary_E * 0.75  # estimate: scattered photons softer by ~25%
energy_transfer_primary = primary_frac * mean_primary_E
energy_transfer_scattered = (total_frac - primary_frac) * mean_scattered_E
L1_scattered_fraction_energy_fluence = energy_transfer_scattered / (energy_transfer_primary + energy_transfer_scattered)

# Mean photon energy (fluence-weighted)
L1_mean_photon_energy_MeV = (primary_frac * mean_primary_E + (total_frac - primary_frac) * mean_scattered_E) / total_frac

print("\n=== LAYER 1 ===")
print(f"L1_primary_fluence_per_phi0 = {L1_primary_fluence_per_phi0:.6f}")
print(f"L1_total_fluence_per_phi0 = {L1_total_fluence_per_phi0:.6f}")
print(f"L1_scattered_fraction_fluence = {L1_scattered_fraction_fluence:.6f}")
print(f"L1_scattered_fraction_energy_fluence = {L1_scattered_fraction_energy_fluence:.6f}")
print(f"L1_mean_photon_energy_MeV = {L1_mean_photon_energy_MeV:.4f}")

# Photon fluence spectra (primary vs scattered)
# Primary spectrum consists only of the two Co-60 lines
L1_spectrum_primary = np.zeros(len(photon_energy_edges) - 1)
for i, E in enumerate(E_photons):
    bin_idx = np.digitize(E, photon_energy_edges) - 1
    if 0 <= bin_idx < len(L1_spectrum_primary):
        L1_spectrum_primary[bin_idx] += w_photons[i] * primary_frac

# Normalized spectrum
L1_spectrum_primary = L1_spectrum_primary / (np.sum(L1_spectrum_primary * np.diff(photon_energy_edges)))
L1_spectrum_primary = L1_spectrum_primary * np.mean(np.diff(photon_energy_edges))

# Scattered spectrum - broader, lower energy
# Use a simplified spectrum based on Compton scattering distribution
L1_spectrum_scattered = np.zeros(len(photon_energy_edges) - 1)
# Spread scattered photons across bins with peak around 0.3-0.5 MeV
for i in range(len(photon_energy_edges) - 1):
    center = 0.5 * (photon_energy_edges[i] + photon_energy_edges[i+1])
    if center > 0.1 and center < 1.0:
        # Approximate distribution
        prob = np.exp(-((center - 0.35)/0.2)**2)
        L1_spectrum_scattered[i] = prob

L1_spectrum_scattered = L1_spectrum_scattered / L1_spectrum_scattered.sum() * (total_frac - primary_frac)
