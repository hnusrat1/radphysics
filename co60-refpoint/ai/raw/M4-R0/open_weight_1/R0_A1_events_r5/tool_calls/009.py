import numpy as np
from scipy.interpolate import interp1d
import json

np.random.seed(42)

# ============================================================
# Constants
# ============================================================
m_e = 0.51099895
r_e = 2.8179403262e-13
N_A = 6.02214076e23

# ============================================================
# Cross sections for water (cm^-1 at density 1.0 g/cm^3)
# ============================================================
e_arr = np.array([1.0, 1.1, 1.1732, 1.2, 1.25, 1.3, 1.3325, 1.4, 1.5])
mu_t = np.array([0.07070, 0.06780, 0.06590, 0.06520, 0.06400, 0.06280, 0.06220, 0.06080, 0.05780])
mu_c = np.array([0.06950, 0.06650, 0.06430, 0.06350, 0.06200, 0.06060, 0.06000, 0.05850, 0.05550])
mu_p = np.array([0.00040, 0.00030, 0.00025, 0.00022, 0.00020, 0.00017, 0.00015, 0.00013, 0.00010])
mu_pp = np.array([0.0, 0.0, 0.00020, 0.00028, 0.00050, 0.00080, 0.00100, 0.00130, 0.00150])
mu_r = np.array([0.00080, 0.00068, 0.00060, 0.00055, 0.00050, 0.00043, 0.00040, 0.00035, 0.00030])

f_t = interp1d(e_arr, mu_t, kind='cubic', bounds_error=False, fill_value=(0.0707, 0.0578))
f_c = interp1d(e_arr, mu_c, kind='cubic', bounds_error=False, fill_value=(0.0695, 0.0555))
f_p = interp1d(e_arr, mu_p, kind='cubic', bounds_error=False, fill_value=(0.0004, 0.0001))
f_pp = interp1d(e_arr, mu_pp, kind='cubic', bounds_error=False, fill_value=(0.0, 0.0015))
f_r = interp1d(e_arr, mu_r, kind='cubic', bounds_error=False, fill_value=(0.0008, 0.0003))

# Oops, got too many typos. Let me start fresh with clean code.