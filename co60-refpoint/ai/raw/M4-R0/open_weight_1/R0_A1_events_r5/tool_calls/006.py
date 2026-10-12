import numpy as np
from scipy.interpolate import interp1d
import json
import sys

np.random.seed(42)

# ============================================================
# Physical constants
# ============================================================
m_e_c2 = 0.51099895  # MeV
r_e = 2.8179403262e-13  # cm
N_A = 6.02214076e23

# ============================================================
# Cross sections for water (NIST-based)
# ============================================================
energy_pts = np.array([1.0, 1.1, 1.1732, 1.2, 1.25, 1.3, 1.3325, 1.4, 1.5])
# Fix syntax error
energy_pts = np.array([ 1.0, 1.1, 1.1732, 1.2, 1.25, 1.3, 1.3325, 1.4, 1.5])