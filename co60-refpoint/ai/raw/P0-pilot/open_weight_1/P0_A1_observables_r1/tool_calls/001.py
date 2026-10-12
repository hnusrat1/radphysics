import numpy as np
import scipy.integrate as integrate
import math

# Physical constants
mec2 = 0.511  # MeV, electron rest mass energy
r0 = 2.818e-13  # cm, classical electron radius
r0_sq = r0**2

print("Constants loaded")

# Let me compute the Klein-Nishna cross section and related quantities

# Energy grid for photons
photon_energy_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
print(f"Number of photon energy bins: {len(photon_energy_edges)-1}")

electron_energy_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.4])
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
# Field size at phantom surface
field_x= 10.0  # cm
field_y = 10.0  # cm

# Distance from source to point of interest
r_poi = SSD + depth  # 110 cm

# Solid angle of the pyramid
# For a square field of side F at distance SSD, the solid angle subtended at the source:
# Using formula: Ω = 4 * aresin(F^2 / (4*SSD^2 + 2*F^2))
# Actually, for a rectangle of half-angles alpha, beta:
# alpha = atan(F/2 / SSD) = atan(5/100)
# beta = same
# Ω = 4 * arcsin(sin(alpha) * sin(beta))
alpha = math.atan(field_x/2 / SSD)
beta = math.atan(field_y/2 / SSD)
# Solid angle for pyramid
# Ω = 4 * arcsn(sin(alpha)*sin(beta))
# Actually, the solid angle of a pyramid with rectangular base:
omega_pyramid = 4 * math.asin( math.sin(alpha) * math.sin(beta) )
print(f"Solid angle of pyramid: {omega_pyramid:.6f} sr")
print(f"Check: area/SSD^2 = {field_x*field_y/SSD**2:.6f} sr")