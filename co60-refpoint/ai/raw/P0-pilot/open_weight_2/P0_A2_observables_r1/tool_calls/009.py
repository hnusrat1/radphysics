import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

# Read data files
xcom_data = []
with open('/tmp/data/xcom_water.txt', 'r') as f:
    for line in f:
        if not line.startswith('#'):
            parts = [float(x) for x in line.split()]
            xcom_data.append(parts)
xcom = np.array(xcom_data)

estar_data = []
with open('/tmp/data/estar_water.txt', 'r') as f:
    for line in f:
        if not line.startswith('#'):
            parts = [float(x) for x in line.split()]
            estar_data.append(parts)
estar = np.array(estar_data)

# Parameters
E_photon = 2.0  # MeV
rho = 1.0  # g/cm^3
depth = 10.0  # cm
mu_total_mass = xcom[28, 7]  # total without coherent at 2.0 MeV
mu_compton_mass = xcom[28, 2]
mu_pair_nuc_mass = xcom[28, 4]
mu_pair_el_mass = xcom[28, 5]
mu_rayleigh_mass = xcom[28, 1]
mu_photo_mass = xcom[28, 3]

mu_total = rho * mu_total_mass
mu_compton = rho * mu_compton_mass
mu_pair = rho * (mu_pair_nuc_mass + mu_pair_el_mass)
mu_rayleigh = rho * mu_rayleigh_mass
mu_photo = rho * mu_photo_mass

# Bins
photon_energy_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                                0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 
                                1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
electron_energy_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 
                                  1.1, 1.15, 1.2, 1.4])
cos_theta_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

n_photon_bins = len(photon_energy_edges) - 1
n_electron_bins = len(electron_energy_edges) - 1
n_cos_bins = len(cos_theta_edges) - 1

print("Layer 1: Photons arriving at the point")
print("="*50)

# L1_primary_fluence_per_phi0
L1_primary_fluence_per_phi0 = np.exp(-mu_total * depth)
print(f"L1_primary_fluence_per_phi0 = {L1_primary_fluence_per_phi0:.6f}")

# L1_photon_fluence_spectrum_primary (all primary photons at 2.0 MeV)
L1_photon_fluence_spectrum_primary = np.zeros(n_photon_bins)
for i in range(n_photon_bins):
    if photon_energy_edges[i] <= E_photon < photon_energy_edges[i+1]:
        L1_photon_fluence_spectrum_primary[i] = L1_primary_fluence_per_phi0
        break

# L1_angular_primary (all forward, cos(theta)=1)
L1_angular_primary = np.zeros(n_cos_bins)
L1_angular_primary[n_cos_bins - 1] = 1.0

# Scattered photon calculations
# Estimate scattered fluence from single Compton scattering along the path
# At each depth z, probability of Compton interaction per unit length is mu_compton
# Scattered photons then travel to depth d=10 cm with additional attenuation

# First-order approximation: scattered photons come from single scatterings along path
P_scatter_est = 0  # Will be estimated below

# For estimation, assume roughly uniform interaction rate along path
# Average path length to first interaction ~ 1/mu_total = 55.2 cm
# At 10 cm depth, fraction of photons interacting = 1 - exp(-0.0493 * 10) = 0.389

alpha = E_photon / 0.511

def klein_nishina_intensity(alpha, ct):
    k = 1.0 + alpha * (1.0 - ct)
    sin_sq = 1.0 - ct**2
    return (1.0/k**2) * (k + 1.0/k - sin_sq)

def scattered_energy_from_KN(ct):
    k = 1.0 + alpha * (1.0 - ct)
    return E_photon / k

def electron_energy_from_KN(ct):
    return E_photon - scattered_energy_from_KN(ct)

# Integrate over cos(theta) from 0 to 1 (forward scattered mostly)
# Actually, let's integrate from -1 to 1 for full distribution
num_int, _ = quad(lambda ct: scattered_energy_from_KN(ct) * klein_nishina_intensity(alpha, ct), -1, 1)
denom_int, _ = quad(lambda ct: klein_nishina_intensity(alpha, ct), -1, 1)
mean_scattered_E_photon = num_int / denom_int
fraction_E_transfer = (E_photon - mean_scattered_E_photon) / E_photon

print(f"\nMean scattered photon energy (Compton): {mean_scattered_E_photon:.4f} MeV")
print(f"Fractional energy transfer: {fraction_E_transfer:.4f}")

# Estimate total scattered photon fluence
# Rough estimate: most scatterings are single scatters
# Number of scattered photons produced = integral of mu_compton * Phi(z) dz from 0 to d
# where Phi(z) = Phi0 * exp(-mu_total * z)
# This gives N_scattered = Phi0 * (mu_compton / mu_total) * (1 - exp(-mu_total * d))
# But these scattered photons then attenuate...
# Approximate average attenuation factor for scattered photons ~ exp(-mu_total * d/2) since they're produced at various depths

P_interact_total = 1 - np.exp(-mu_total * depth)
f_compton_frac = mu_compton / mu_total
est_scattered_fraction = P_interact_total * f_compton_frac * 0.5  # rough avg attenuation factor

L1_scattered_fraction_fluence = est_scattered_fraction
L1_total_fluence_per_phi0 = L1_primary_fluence_per_phi0 + L1_scattered_fraction_fluence

print(f"P(interaction before depth d): {P_interact_total:.4f}")
print(f"Estimate of L1_scattered_fraction_fluence: {L1_scattered_fraction_fluence:.4f}")
print(f"L1_total_fluence_per_phi0: {L1_total_fluence_per_phi0:.4f}")

# L1_mean_photon_energy_MeV (fluence-weighted)
mean_E_primary = E_photon  # all primary at 2.0 MeV
mean_E_scattered = mean_scattered_E_photon  # approximate

if L1_total_fluence_per_phi0 > 0:
    L1_mean_photon_energy_MeV = (L1_primary_fluence_per_phi0 * mean_E_primary + 
                                 L1_scattered_fraction_fluence * mean_E_scattered) / L1_total_fluence_per_phi0
else:
    L1_mean_photon_energy_MeV = E_photon
print(f"L1_mean_photon_energy_MeV: {L1_mean_photon_energy_MeV:.4f}")

# L1_scattered_fraction_energy_fluence
energy_fluence_primary = L1_primary_fluence_per_phi0 * mean_E_primary
energy_fluence_scattered = L1_scattered_fraction_fluence * mean_E_scattered
total_energy_fluence = energy_fluence_primary + energy_fluence_scattered
L1_scattered_fraction_energy_fluence = energy_fluence_scattered / total_energy_fluence if total_energy_fluence > 0 else 0
print(f"L1_scattered_fraction_energy_fluence: {L1_scattered_fraction_energy_fluence:.4f}")