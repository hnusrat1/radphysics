import json
import numpy as np

# Re-calculate L5 properly
E_avg = 1.25 # MeV
psi_per_phi0 = E_avg * 1.60217662e-13 # J/cm^2 per particle/cm^2
mu_en_rho = 0.029 # cm^2/g
attenuation = 0.7269
D_per_phi0 = psi_per_phi0 * mu_en_rho * 1000 * attenuation # J/kg = Gy
print("D/phi0 =", D_per_phi0)

beta = 0.98
g = 0.0004