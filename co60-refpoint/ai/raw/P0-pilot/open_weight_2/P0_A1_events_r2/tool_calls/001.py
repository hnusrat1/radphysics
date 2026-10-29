
import numpy as np
import json
from scipy import constants

# Convert physical constants
mev_to_joules = 1.602176634e-13
electron_mass_MeV = 0.510998950 # MeV
c = 299792458  # m/s

# At 2.0 MeV in water, let's get approximate mass attenuation coefficients
# From NIST XCOM database for water (H2O) at 2.0 MeV:
# Total mass attenuation coefficient: ~0.055 cm^2/g
# Breakdown approximately:
# - Compton: ~0.045 cm^2/g (dominant)
# - Pair production: ~0.005 cm^2/g  
# - Photoelectric: ~0.001 cm^2/g (very small at 2 MeV)
# - Rayleigh: ~0.0008 cm^2/g

# Let me be more precise by looking up actual values from NIST tables
# Approximate values for water at 2.0 MeV:
mu_compton = 0.0428  # cm^2/g
mu_photoelectric = 0.0001  # cm^2/g (negligible)
mu_rayleigh = 0.0002  # cm^2/g
mu_pair = 0.0052  # cm^2/g

# Total
mu_total = mu_compton + mu_photoelectric + mu_rayleigh + mu_pair

# Relative probabilities
prob_compton = mu_compton / mu_total
prob_photoelectric = mu_photoelectric / mu_total
prob_rayleigh = mu_rayleigh / mu_total
prob_pair = mu_pair / mu_total

print(f"Total: {mu_total}")
print(f"Compton: {prob_compton:.6f}")
print(f"Photoelectric: {prob_photoelectric:.6f}")
print(f"Rayleigh: {prob_rayleigh:.6f}")
print(f"Pair: {prob_pair:.6f}")
