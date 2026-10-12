import numpy as np
from scipy.interpolate import interp1d
import json

np.random.seed(42)

# ============================================================
# Physical constants
# ================================================--====
m_e_c2 = 0.51099895  # MeV
r_e = 2.8179403262e-13  # cm
N_A = 6.02214076e23

# ============================================================
# Cross sections for water (NIST-based, interolated)
# ============================================================
energies = np.array([1.0, 1.1, 1.1732, 1.2, 1.25, 1.3, 1.3325, 1.4, 1.5])
mu_tot = np.array([0.07070, 0.06780, 0.06590, 0.06520, 0.06400, 0.06280, 0.06220, 0.06080, 0.05780])
mu_comp = np.array([0.06950, 0.06650, 0.06430, 0.06350, 0.06200, 0.06060, 0.06000, 0.05850, 0.05550])
mu_photo = np.array([0.00040, 0.00030, 0.00025, 0.00022, 0.00020, 0.00017, 0.00015, 0.00013, 0.00010])
mu_pair = np.array([0.0, 0.0, 0.00020, 0.00028, 0.00050, 0.00080, 0.00100, 0.00130, 0.00150])
mu_rayl = np.array([0.00080, 0.00068, 0.00060, 0.00055, 0.00050, 0.00043, 0.00040, 0.00035, 0.00030])

f_total = interp1d(energies, mu_tot, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_comp = interp1d(energies, mu_comp, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_photo = interp1d(energies, mu_photo, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_pair = interp1d(energies, mu_pair, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_rayl = interp1d(energies, mu_rayl, kind='cubic', bounds_error=False, fill_value='extrapolate')

def mu_total(E): return float(f_total(np.clip(E, 0.01, 10.0)))
def mu_comp(E): return float(f_comp(np.clip(E, 0.01, 10.0)))
def mu_photo(E): return float(f_photo(np.clip(E, 0.01, 10.0)))
def mu_pair(E): return float(f_pair(np.clip(E, 0.01, 10.0)))
def mu_rayl(E): return float(f_rayl(np.clip(E, 0.01, 10.0)))

# Geometry
SSD = 100.0
field_half = 5.0
z_surface = 0.0
z_core = 5.0
scoring_half = 0.1
z_min = z_core - scoring_half
z_max = z_core + scoring_half
scoring_r = 1.0
src = np.array([0.0, 0.0, -SSD])

# Solid angle of pyramid
theta_half = np.arctan(field_half / SSD)
Omega_pyr = 4 * np.arcsin(np.sin(theta_half)**2)
print(f"Pyramid solid angle: {Omega_pyr:.6f} sr")

# Electron density of water
n_e = 10 * N_A * 1.0 / 18.015

# ============================================================
# Physics functions
# ============================================================
E_print("Setting up physics...")