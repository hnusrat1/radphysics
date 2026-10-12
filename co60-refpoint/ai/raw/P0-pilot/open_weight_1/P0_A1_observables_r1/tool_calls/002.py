import numpy as np
import scipy.integrate as integrate
import math

# Physical constants
mec2 = 0.511  # MeV, electron rest mass energy

# Energy grid for photons
photon_energy_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
print(f"Number of photon energy bins: {len(photon_energy_edges)-1}")

electron_energy_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.4])
print(f"Number of electron energy bins: {len(electron_energy_edges)-1}")

cotheta_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
print(f"Number of cos(theta) bins: {len(cotheta_edges)-1}")

# Source energy
E0 = 2.0  # MeV

# Water density
rho = 1.0  # g/cm^3

# Geometry
SSD = 100.0  # cm
depth = 10.0  # cm
field_x= 10.0  # cm
field_y = 10.0  # cm

# Distance from source to point of interest
r_poi = SSD + depth  # 110 cm

# Solid angle of the pyramid
alpha = math.atan(field_x/2 / SSD)
beta = math.atan(field_y/2 / SSD)
omega_pyramid = 4 * math.asin( math.sin(alpha) * math.sin(beta) )
print(f"Solid angle of pyramid: {omega_pyramid:.6f} sr")
print(f"Check: area/SSD^2 = {field_x*field_y/SSD**2:.6f} sr")