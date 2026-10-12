import numpy as np
import json

# ===== LOAD DATA =====
xcom = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
estar = np.loadtxt('/tmp/data/estar_water.txt', comments='#')

def interp_xcom(col_idx, E):
    return np.interp(np.log10(E), np.log10(xcom[:, 0]), xcom[:, col_idx])

def interp_estar(col_idx, E):
    return np.interp(E, estar[:, 0], estar[:, col_idx])

# ===== PARAMETERS =====
E_photons = np.array([1.1732, 1.3325])
w_photons = np.array([0.5, 0.5])
SSD = 100.0  # cm
depth = 5.0  # cm  
rho = 1.0  # g/cm^3

# Bins
photon_energy_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                                0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
                                1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
electron_energy_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                                  0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                                  1, 1.05, 1.1, 1.15, 1.2, 1.4])
cos_theta_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

# ===== LAYER 1 CALCULATIONS =====

# Primary transmission
mu_mass_total_w = sum(w * interp_xcom(6, E) for w, E in zip(w_photons, E_photons))
mu_lin = mu_mass_total_w * rho  # cm^-1
primary_frac = np.exp(-mu_lin * depth)

# Buildup estimate
scatter_to_primary = 0.18
total_frac = primary_frac * (1 + scatter_to_primary)

L1_primary = primary_frac
L1_total = total_frac
L1_scatter_frac_fluence = 1 - L1_primary / L1_total

# Mean energies
mean_E_primary = np.average(E_photons, weights=w_photons)
mean_E_scattered = mean_E_primary * 0.78
L1_mean_E = (L1_primary * mean_E_primary + (L1_total - L1_primary) * mean_E_scattered) / L1_total

# Energy fluence
efluo_primary = L1_primary * mean_E_primary
efluo_scattered = (L1_total - L1_primary) * mean_E_scattered
L1_scatter_frac_efluo = efluoa_scattered / (efluoa_primary + efluoa_scattered) if 'efluoa_scattered' not in locals() else efluoa_scattered / (efluoa_primary + efluoa_scattered)

print("Layer 1 values calculated")
print(f"L1_primary = {L1_primary:.6f}")
print(f"L1_total = {L1_total:.6f}")
print(f"L1_scatter_frac_fluence = {L1_scatter_frac_fluence:.6f}")
print(f"L1_mean_E = {L1_mean_E:.4f}")
