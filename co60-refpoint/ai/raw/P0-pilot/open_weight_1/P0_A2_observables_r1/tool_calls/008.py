import numpy as np
from scipy.interpolate import interp1d

# Load data
xcom_data = np.loadtxt('/tmp/data/xcom_water.txt')
xcom_energies = xcom_data[:, 0]
xcom_coherent = xcom_data[:, 1]
xcom_incoherent = xcom_data[:, 2]
xcom_photo = xcom_data[:, 3]
xcom_pair_nuc = xcom_data[:, 4]
xcom_pair_elec = xcom_data[:, 5]
xcom_total = xcom_data[:, 6]

estar_data = np.loadtxt('/tmp/data/estar_water.txt')
estar_energies = estar_data[:, 0]
estar_coll = estar_data[:, 1]
estar_rad = estar_data[:, 2]
estar_total_sp = estar_data[:, 3]
estar_csda = estar_data[:, 4]
estar_rad_yield = estar_data[:, 5]
estar_delta = estar_data[:, 6]

def interp_xcom(E, y):
    return np.interp(E, xcom_energies, y)

def interp_estar(E, y):
    return np.interp(E, estar_energies, y)

# Constants
m_e_c2 = 0.511
rho = 1.0  # g/cm^3

# Geometry
SSD = 100.0  # cm
field_size = 10.0  # cm x cm at phantom surface
depth_POI = 10.0  # cm
scoring_radius = 1.0  # cm
scoring_half_thick = 0.1  # cm (half of 0.2 cm)

# Source position: on central axis at z = -SSD, where z=0 is phantom surface
# Phantom extends from z=0 to z=30 cm
# Point of interest at z = 10 cm

# Primary photon energy
E0 = 2.0  # MeV

# Cross sections at 2.0 MeV
mu_tot_0 = interp_xcom(E0, xcom_total)
mu_inc_0 = interp_xcom(E0, xcom_incoherent)
mu_coh_0 = interp_xcom(E0, xcom_coherent)
mu_photo_0 = interp_xcom(E0, xcom_photo)
mu_pair_0 = interp_xcom(E0, xcom_pair_nuc) + interp_xcom(E0, xcom_pair_elec)

print(f"Cross sections at {E0} MeV (cm^2/g):")
print(f"  Total: {mu_tot_0:.6f}")
print(f"  Incoherent: {mu_inc_0:.6f}")
print(f"  Coherent: {mu_coh_0:.6e}")
print(f"  Photo: {mu_photo_0:.6e}")
print(f"  Pair: {mu_pair_0:.6e}")

# Attenuation coefficients (cm^-1)
mu_tot_0_cm = mu_tot_0 * rho
mu_inc_0_cm = mu_inc_0 * rho
mu_coh_0_cm = mu_coh_0 * rho
mu_photo_0_cm = mu_photo_0 * rho
mu_pair_0_cm = mu_pair_0 * rho

print(f"\nAttenuation coefficients (cm^-1):")
print(f"  Total: {mu_tot_0_cm:.6f}")


# Primary photon fluence at scoring volume / Ph0
# At scoring volue (at depth 10 cm, 110 cm from source)
# The primar photons are atteuated through 10 cm of water
L1_primary_fluence_per_phi0 = np.exp(-mu_tot_0 * rho * 10.0)
print(f"\nL1_primary_fluence_per_phi0 = {L1_primary_fluence_per_phi0:.6f}")