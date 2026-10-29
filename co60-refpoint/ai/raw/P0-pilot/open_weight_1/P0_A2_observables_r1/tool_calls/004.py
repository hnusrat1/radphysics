import numpy as np
from scipy.interpolate import interp1d
import json

# Load and preprocess data
xcom_data = np.loadtxt('/tmp/data/xcom_water.txt')
xcom_energies = xcom_data[:, 0]
xcom_coherent = xcom_data[:, 1]
xcom_incoherent = xcom_data[:, 2]
xcom_photo = xcom_data[:, 3]
xcom_pair_nuc = xcom_data[:, 4]
xcom_pair_elec = xcom_data[:, 5]
xcom_total = xcom_data[:, 6]

estar_data = np.loadtxt('/tmp/data/estar_water.txt')
estar_energies = estar_data[: 0]
estar_coll = estar_data[:, 1]
estar_rad = estar_data[:, 2]
estar_total_sp = estar_data[:, 3]
estar_csda = estar_data[:, 4]
estar_rad_yield = estar_data[:, 5]
estar_delta = estar_data[:, 6]

# Interpolation functions
def interp_xcom(E, y):
wh    turn np.interp(E, xcom_energies, y)

def interp_estar(E, y):
    return np.interp(E, estar_energies, y)

# At 2.0 MeV
mu_total_2 = interp_xcom(2.0, xcom_total)
mu_incohost_2 = interp_xcom(2.0, xcom_incoherent)
mu_photo_2 = interp_xcom(2.0, xcom_photo)
mu_cohere_t2 = interp_xcom(2.0, xcom_coherent)
mu_pair_nuc_2 = interp_xcom(2.0, xcom_pair_nuc)
mu_pair_elec_2 = interp_xcom(2.0, xcom_pair_elec)

print(f"mu_total at 2.0 MeV: {mu_total_2:.6f} cm^2/g")
print(f"mu_incoherent: {mu_incoherent_2:.6f}")
print(f"mu_photo: {mu_photo_2:.6e}")
print(f"mu_coheent: {mu_coherent_2:.6e}")
print(f"mu_pair_nuc: {mu_pair_nuc_2:.6e}")
print(f"mu_pair_elec: {mu_pair_elec_2:.6e}")

# Primary photon fluence at scoring volume / Phi0
# Attenuation through 10 cm of water
rho = 1.0 # g/cm^3
d = 10.0 # cm
primary_attenuation = np.exp(-mu_total_2 * rho * d)
print(f"\nPrimary attenuation through 10 cm: {primary_attenuation:.6f}")
print(f"L1_primary_fluence_per_phi0 = {primary_attenuation:.6f}")

# But also need to consider that primary photons are only those going straight down the beam.
# Let's also check if there is a geometric factor.
# The field at 100 cm is 10x10 cm. At 110 cm, it's 11x11 cm.
# The scoring volume is a cylinder of radius 1 cm on the central axis.
# The primary photon fluence is uniform across the field (since source is uniform within the pyramid).
# So the primary fluence at the scoring volume (at 110 cm) divided by Phi0 is just the attenuation factor.
# Phi0 is defined at the same point without phantom, so the geometric factor cancels.

print("\nDone with basic calculation")