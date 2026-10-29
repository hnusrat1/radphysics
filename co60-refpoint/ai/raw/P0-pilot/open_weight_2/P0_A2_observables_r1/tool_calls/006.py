import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

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

# Define parameters
E_photon = 2.0  # MeV
rho = 1.0  # g/cm^3
depth = 10.0  # cm
r_source_surf = 100.0  # cm
r_source_PI = 110.0  # cm

# Bin definitions
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

# Cross-sections at 2.0 MeV
E_idx = 28  # index for 2.0 MeV
mu_rayleigh_mass = xcom[E_idx, 1]
mu_compton_mass = xcom[E_idx, 2]
mu_photo_mass = xcom[E_idx, 3]
mu_pair_nuc_mass = xcom[E_idx, 4]
mu_pair_el_mass = xcom[E_idx, 5]
mu_total_mass = xcom[E_idx, 7]  # without coherent

mu_rayleigh = rho * mu_rayleigh_mass
mu_compton = rho * mu_compton_mass
mu_photo = rho * mu_photo_mass
mu_pair = rho * (mu_pair_nuc_mass + mu_pair_el_mass)
mu_total = rho * mu_total_mass

print("Layer 1 Calculations:")
print("="*50)

# L1_primary_fluence_per_phi0
primary_attenuation = np.exp(-mu_total * depth)
L1_primary_fluence_per_phi0 = primary_attenuation
print(f"L1_primary_fluence_per_phi0 = {L1_primary_fluence_per_phi0:.6f}")

# Primary fluence spectrum
# All primary photons are at 2.0 MeV
L1_photon_fluence_spectrum_primary = np.zeros(n_photon_bins)
for i in range(n_photon_bins):
    e_low, e_high = photon_energy_edges[i], photon_energy_edges[i+1]
    if e_low <= E_photon < e_high:
        L1_photon_fluence_spectrum_primary[i] = L1_primary_fluence_per_phi0

# For angular distribution of primary photons
# The source emits into a pyramid defining a 10x10 cm field at 100 cm
# At the point of interest (110 cm), we're on the central axis
# Due to geometry, primary photons arrive nearly parallel to beam axis
# The angular spread from a point source at distance r through a square aperture is small
# Let's calculate the maximum angle
half_field = 5.0  # cm
max_angle = np.arctan(half_field / r_source_PI)
print(f"Maximum angle of primary photons: {np.degrees(max_angle):.4f} degrees")

# Since max_angle is very small (~2.6 degrees), most photons have cos(theta) close to 1
# For simplicity, assume all primary photons have cos(theta) = 1 (within the scoring volume on axis)
L1_angular_primary = np.zeros(n_cos_bins)
# Find which bin contains cos(theta) = 1
cos_theta_max_bin = n_cos_bins - 1  # last bin
L1_angular_primary[cos_theta_max_bin] = 1.0
print(f"L1_angular_primary: {L1_angular_primary}")

# Scattered photon fluence
# Estimate from single Compton scattering events along the path
# Probability of interaction before depth d: P_interact = 1 - exp(-mu_total * d) ≈ mu_total * d (for small d/lambda)
P_interact = 1 - primary_attenuation
print(f"\nP(interaction before 10 cm) = {P_interact:.6f}")

# Fraction of interactions that are Compton (produce scattered photons)
f_compton = mu_compton / mu_total
f_pair = mu_pair / mu_total
f_rayleigh = mu_rayleigh / mu_total

print(f"f_Compton = {f_compton:.6f}")
print(f"f_Pair = {f_pair:.6f}")
print(f"f_Rayleigh = {f_rayleigh:.6f}")

# For scattered photons, they come mainly from:
# 1. Compton scattering (dominant)
# 2. Bremsstrahlung from electrons produced by pair production
# 3. Annihilation photons (511 keV) from positron annihilation
# 4. Fluorescence (negligible for water)

# For first-order approximation, consider only single Compton scattering
# The scattered photon spectrum depends on the Klein-Nishina differential cross-section

# For the energy fluence from Compton scattering at initial energy E0:
# The average energy transferred to electrons per Compton interaction:
# <T_e> = integral of T * dsigma/dOmega dOmega / sigma_Compton

# For 2.0 MeV photons, using Klein-Nishina formula:
# Maximum electron energy (Compton edge):
alpha = E_photon / 0.511  # ratio of photon energy to electron rest mass
T_max = E_photon * (1 - 1/(1 + 2*alpha))
print(f"\nCompton edge (max electron energy): {T_max:.4f} MeV")

# Average energy transfer for Compton scattering at 2.0 MeV
# This requires integration over angles...
# For alpha = 2.0/0.511 = 3.914, the fractional energy transfer is substantial

# Let me estimate using numerical integration of Klein-Nishina
def klein_nishina_differential(alpha, cos_theta):
    """Differential cross-section (normalized) for Compton scattering"""
    k = 1.0 + alpha * (1.0 - cos_theta)
    return 0.5 * (1.0/k**2) * (k + 1.0/k - sin(1)**2) 

Actually, let me use the proper Klein-Nishina formula.