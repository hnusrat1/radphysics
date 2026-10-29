import numpy as np
from scipy.interpolate import interp1d
import time

# Load and interpolate
xcom_data = np.loadtxt('/tmp/data/xcom_water.txt')
xcom_en = xcom_data[:, 0]
xcom_inc = xcom_data[:, 2]
xcom_coh = xcom_data[:, 1]
xcom_photo = xcom_data[:, 3]
xcom_pair_n = xcom_data[:, 4]
xcom_pair_e = xcom_data[:, 5]
xcom_tot = xcom_data[:, 6]

estar_data = np.loadtxt('/tmp/data/estar_water.txt')
estar_en = estar_data[:, 0]
estar_coll = estar_data[:, 1]
estar_rad = estar_data[:, 2]
estar_csda = estar_data[:, 3]
estar_rad_yield = estar_data[:, 4]

f_inc = interp1d(xcom_en, xcom_inc, kind='linear', bounds_error=False, fill_value=0.0)
f_coh = interp1d(xcom_en, xcom_coh, kind='linear', bounds_error=False, fill_value=0.0)
f_photo = interp1d(xcom_en, xcom_photo, kind='linear', bounds_error=False, fill_value=0.0)
f_pair_n = interp1d(xcom_en, xcom_pair_n, kind='linear', bounds_error=False, fill_value=0.0)
f_pair_e = interp1d(xcom_en, xcom_pair_e, kind='linear', bounds_error=False, fill_value=0.0)
f_tot = interp1d(xcom_en, xcom_tot, kind='linear', bounds_error=False, fill_value=0.0)
f_coll = interp1d(estar_en, estar_coll, kind='linear', bounds_error=False, fill_value=0.0)
f_rad = interp1d(estar_en, estar_rad, kind='linear', bounds_error=False, fill_value=0.0)
f_csda = interp1d(estar_en, estar_csda, kind='linear', bounds_error=False, fill_value=0.0)
f_rad_yield = interp1d(estar_en, estar_rad_yield, kind='linear', bounds_error=False, fill_value=0.0)

mec2 = 0.511
rho = 1.0
E0 = 2.0
SSD = 100.0

# Scoring cylinder
scoring_r = 1.0
scoring_z_min = 9.9
scoring_z_max = 10.1
scoring_vol = np.pi * scoring_r**2 * 0.2

# Bins
photon_bins = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
electron_bins = np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
cos_theta_bins = np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])

# Klein-Nishina differential cross section
# d_sigma/d_Omega = (r_e^2/2) * (E'/E0)^2 * (E0/E' + E'/E0 - sin^2(theta))
# Normalized to 1 at theta=0 (forward scattering) for convenience
# Actually we need the absolute cross section

# Number of electrons per gram in water
# H2O: 10 electrons/molecule, 18.015 g/mol
NA = 6.02214076e23
Ne_per_g = NA * 10 / 18.015  # electrons per gram

# Klein-Nishina cross section per electron (cm^2)
# d_sigma/d_Omega = r_e^2/2 * (E'/E0)^2 * (E0/E' + E'/E0 - sin^2(theta))
r_e = 2.817940e-13  # cm

def compton_scattered_energy(E, theta):
    """Energy of photon scattered by angle theta (radians)"""
    return E / (1 + (E/mec2) * (1 - np.cos(theta)))

def kn_dsigma_dOmega(E, theta):
    """Klein-Nishina differential cross section per electron (cm^2/sr)"""
    E_prime = compton_scattered_energy(E, theta)
    ratio = E_prime / E
    sin2 = np.sin(theta)**2
    return (r_e**2 / 2) * ratio**2 * (1/ratio + ratio - sin2)

def compton_electron_energy(E, theta):
    """Kinetic energy of recoil electron"""
    E_prime = compton_scattered_energy(E, theta)
    return E - E_prime

def compton_electron_cos_theta(E, theta):
    """cos(theta_e) where theta_e is the electron recoil angle relative to incident photon direction"""
    E_prime = compton_scattered_energy(E, theta)
    # cos(theta_e) = (E - E_prime * cos(theta)) / sqrt(E^2 + E_prime^2 - 2*E*E_prime*cos(theta))
    # Actually, from kinematics:
    # tan(theta_e) = cot(theta/2) / (1 + E/mec2)
    # Or: cot(theta_e) = (1 + E/mec2) * tan(theta/2)
    # cos(theta_e) = ?
    # Let me use: cos(theta_e) = (E - E_prime * cos(theta)) / sqrt(E^2 + E_prime^2 - 2*E*E_prime*cos(theta))
    # Actually, from conservation of momentum:
    # p_e * cos(theta_e) = E/c - E'/c * cos(theta)  [for photon momentum E/c]
    # p_e * sin(theta_e) = E'/c * sin(theta)
    # So tan(theta_e) = E' * sin(theta) / (E - E' * cos(theta))
    tan_theta_e = E_prime * np.sin(theta) / (E - E_prime * np.cos(theta))
    theta_e = np.arctan2(E_prime * np.sin(theta), E - E_prime * np.cos(theta))
    # Make sure theta_e is in [0, pi/2] for Compton
    return np.cos(theta_e)

# Test Compton kinematics
print("Testing Compton kinematics at E0=2.0 MeV")
for theta_deg in [0, 30, 60, 90, 120, 150, 180]:
    theta = np.radians(theta_deg)
    E_p = compton_scattered_energy(E0, theta)
    E_e = compton_electron_energy(E0, theta)
    cos_e = compton_electron_cos_theta(E0, theta)
    print(f"  theta={theta_deg:3d}°: E'={E_p:.4f} MeV, E_e={E_e:.4f} MeV, cos(theta_e)={cos_e:.4f}")

print("\nKinematics test done")