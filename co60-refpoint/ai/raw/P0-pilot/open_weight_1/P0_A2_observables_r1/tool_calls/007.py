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

print("Data loaded successfully")

# Physical constants
m_e_c2 = 0.511  # MeV
r_e = 2.818e-13  # cm

# Photon energy bins
photon_bins = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
photon_centers = (photon_bins[1:] + photon_bins[:-1]) / 2

# Electron energy bins
electron_bins = np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
electron_centers = (electron_bins[1:] + electron_bins[:-1]) / 2

# cos_theta bins
cos_theta_bins = np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
cos_theta_centers = (cos_theta_bins[1:] + cos_theta_bins[:-1]) / 2

print("Bins defined")
print(f"Photon bins: {len(photon_bins)-1}")
print(f"Electron bins: {len(electron_bins)-1}")
print(f"Cos theta bins: {len(cos_theta_bins)-1}")

# Fix: there's a typo in electron_bins - 1.4 is the last edge
# Actually let me re-check
print("\nElectron bins edges:")
print(electron_bins)
print(f"Number of electron bins: {len(electron_bins)-1}")