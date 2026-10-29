import numpy as np
from scipy.interpolate import interp1d
import json

np.random.seed(42)

# === Constants ===
m_e = 0.51099895  # MeV
r_e = 2.8179403262e-13  # cm
NA = 6.02214076e23

# === Cross sections for water (linear attenuation, cm^-1, density=1 g/cm^3) ===
e_arr = np.array([1.0, 1.1, 1.1732, 1.2, 1.25, 1.3, 1.3325, 1.4, 1.5])