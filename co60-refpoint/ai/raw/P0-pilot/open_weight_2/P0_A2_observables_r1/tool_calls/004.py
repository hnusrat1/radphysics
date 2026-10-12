import numpy as np
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

# Define parameters
E_photon = 2.0  # MeV
rho = 1.0  # g/cm^3
depth = 10.0  # cm
r_source_surf = 100.0  # cm
r_source_PI = 110.0  # cm
field_size = 10.0  # cm

# Bin definitions
photon_energy_edges = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                       0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 
                       1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]
electron_energy_edges = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                         0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 
                         1.1, 1.15, 1.2, 1.4]
cos_theta_edges = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]

n_photon_bins = len(photon_energy_edges) - 1
n_electron_bins = len(electron_energy_edges) - 1
n_cos_bins = len(cos_theta_edges) - 1

print(f"Photon bins: {n_photon_bins}, Electron bins: {n_electron_bins}, Cos bins: {n_cos_bins}")

# Find cross-sections at 2.0 MeV
E_idx = np.argmin(np.abs(xcom[:, 0] - E_photon))
print(f"\nCross-sections at 2.0 MeV:")
print(f"  Coherent (Rayleigh): {xcom[E_idx, 1]:.6e} cm^2/g")
print(f"  Incoherent (Compton): {xcom[E_idx, 2]:.6e} cm^2/g")
print(f"  Photoelectric: {xcom[E_idx, 3]:.6e} cm^2/g")
print(f"  Pair nuclear: {xcom[E_idx, 4]:.6e} cm^2/g")
print(f"  Pair electron: {xcom[E_idx, 5]:.6e} cm^2/g")
print(f"  Total (with coherent): {xcom[E_idx, 6]:.6e} cm^2/g")
print(f"  Total (without coherent): {xcom[E_idx, 7]:.6e} cm^2/g")

# Linear attenuation coefficients (cm^-1)
mu_rayleigh = rho * xcom[E_idx, 1]
mu_compton = rho * xcom[E_idx, 2]
mu_photo = rho * xcom[E_idx, 3]
mu_pair = rho * (xcom[E_idx, 4] + xcom[E_idx, 5])
mu_total = rho * xcom[E_idx, 7]  # without coherent is essentially same

print(f"\nLinear attenuation coefficients (cm^-1):")
print(f"  Rayleigh: {mu_rayleigh:.6e}")
print(f"  Compton: {mu_compton:.6e}")
print(f"  Photoelectric: {mu_photo:.6e}")
print(f"  Pair: {mu_pair:.6e}")
print(f"  Total: {mu_total:.6e}")

# Attenuation at 10 cm
attenuation = np.exp(-mu_total * depth)
print(f"\nPrimary attenuation factor at {depth} cm: {attenuation:.6f}")

# Interaction fractions (by cross-section)
total_cross = mu_total / rho
f_compton = mu_compton / mu_total
f_photo = mu_photo / mu_total
f_pair = mu_pair / mu_total
f_rayleigh = mu_rayleigh / mu_total

print(f"\nInteraction fractions:")
print(f"  Compton: {f_compton:.6f}")
print(f"  Photoelectric: {f_photo:.6e}")
print(f"  Pair: {f_pair:.6f}")
print(f"  Rayleigh: {f_rayleigh:.6e}")