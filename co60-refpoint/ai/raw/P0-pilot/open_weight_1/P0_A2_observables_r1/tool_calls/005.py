import numpy as np
from scipy.interpolate import interp1d
import json

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

# Interpolation functions
def interp_xcom(E, y):
    return np.interp(E, xcom_energies, y)

def interp_estar(E, y):
    return np.interp(E, estar_energies, y)

# At 2.0 MeV
mu_total_2 = interp_xcom(2.0, xcom_total)
mu_incoherent_2 = interp_xcom(2.0, xcom_incoherent)
mu_photo_2 = interp_xcom(2.0, xcom_photo)
mu_coherent_2 = interp_xcom(2.0, xcom_coherent)
mu_pair_nuc_2 = interp_xcom(2.0, xcom_pair_nuc)
mu_pair_elec_2 = interp_xcom(2.0, xcom_pair_elec)

print(f"mu_total at 2.0 MeV: {mu_total_2:.6f} cm^2/g")
print(f"mu_incoherent: {mu_incoherent_2:.6f}")
print(f"mu_photo: {mu_photo_2:.6e}")
print(f"mu_coherent: {mu_coherent_2:.6e}")
print(f"mu_pair_nuc: {mu_pair_nuc_2:.6e}")
print(f"mu_pair_elec: {mu_pair_elec_2:.6e}")

rho = 1.0  # g/cm^3
d = 10.0   # cm depth

# Primary photon fluence (normalized to Phi0)
primary_attenuation = np.exp(-mu_total_2 * rho * d)
print(f"\nPrimary attenuation: {primary_attenuation:.6f}")
print(f"L1_primary_fluence_per_phi0 = {primary_attenuation:.6f}")

# Check the geometry more carefully
# SSD = 100 cm, field = 10x10 cm at phantom surface
# Point at 10 cm depth, so total distance from source = 110 cm
# The scoring cylinder: radius 1 cm, thickness 0.2 cm, centered at 10 cm depth
# So it extends from 9.9 cm to 10.1 cm depth.

# The primary photons all travel along straight lines from the source.
# At 110 cm from source, the field is 11x11 cm.
# The primary fluence is uniform across the field.
# The scoring cylinder of radius 1 cm is well within the field (half-width 5.5 cm at 110 cm).

print("\nField half-width at scoring depth: ", 5 * 110/100, "cm")
print("Scoring cylinder radius: 1 cm, well within field")

# Let me now compute the primary photon spectrum.
# Primary photons are all at 2.0 MeV (monoenergetic).
# So the spectrum is all in the bin containing 2.0 MeV.
# But the photon energy bins only go up to 1.4 MeV!
# The highest bin edge is1.4 MeV, so 2.0 MeV is above all bins.
# Hmm, that's a problem. Let me re-read the bin edges.

print("\nPhoton energy bins:")
photon_bins = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
print(photon_bins)
print(f"\nNumber of bins: {len(photon_bins)-1}")

# The highest bin is 1.35 to 1.4 MeV. 2.0 MeV is above this.
# So all primary photons would be above the highest bin.
# That means the primary spectrum would have zeros in all bins...
# Unless the binning is meant to capture scattered photons only?
# Or maybe I misread. Let me check the problem statement again.

print("\nThe photon energy bins only go up to 1.4 MeV.")
print("2.0 MeV primary photons are above this range.")
print("This means the primary fluence spectrum will be all zeros in the given bins.")
print("But that can't be right. Let me re-read the problem.")

# Actually, re-reading: "photon_energy (28 bins): 0, 0.05, 0.1, ... 1.35, 1.4"
# So the bins only cover up to 1.4 MeV. Primary photons at 2.0 MeV are above this.
# This is probably intentional - the primary photons would be in an "overflow" bin not listed, or
# the definition says "fluence per energy bin" and if the primary energy is above the binning range,
# it doesn't contribute to any bin. But then L1_photon_fluence_spectrum_primary would be all zeros.
# That seems odd but possible since the scattered photons have lower energies.

# Let me just proceed with the calculation.