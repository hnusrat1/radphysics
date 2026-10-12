import numpy as np
import json

# Load data files
xcom = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
estar = np.loadtxt('/tmp/data/estar_water.txt', comments='#')

def interp_xcom(col_idx, E):
    """Interpolate XCOM column at energy E using log-log interpolation"""
    return np.interp(np.log10(E), np.log10(xcom[:, 0]), xcom[:, col_idx])

def interp_estar(col_idx, E):
    """Interpolate ESTAR column at energy E using linear interpolation"""
    return np.interp(E, estar[:, 0], estar[:, col_idx])

# Co-60 photon energies and weights
E_photons = np.array([1.1732, 1.3325])
w_photons = np.array([0.5, 0.5])

# Geometry parameters
SSD = 100.0  # cm
depth_cm = 5.0  # cm
rho_water = 1.0  # g/cm^3

# Bin definitions
photon_energy_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                                0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
                                1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
electron_energy_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                                  0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                                  1, 1.05, 1.1, 1.15, 1.2, 1.4])
cos_theta_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

print("Starting calculations...")
print(f"XCOM data shape: {xcom.shape}")
print(f"ESTAR data shape: {estar.shape}")

# Layer 1: Primary attenuation
mu_mass_total_w = sum(w * interp_xcom(6, E) for w, E in zip(w_photons, E_photons))
mu_lin_total = mu_mass_total_w * rho_water
primary_transmission = np.exp(-mu_lin_total * depth_cm)

print(f"Primary transmission through {depth_cm} cm water: {primary_transmission:.6f}")
print(f"Mass attenuation coefficient (weighted): {mu_mass_total_w:.6e} cm^2/g")
