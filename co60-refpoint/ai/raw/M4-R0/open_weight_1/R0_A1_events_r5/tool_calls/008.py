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
# Cross sections (NIST-based for water, linear attenuation cm^-1)
# ============================================================
e_arr = np.array([1.0, 1.1, 1.1732, 1.2, 1.25, 1.3, 1.3325, 1.4, 1.5])
mu_t = np.array([0.07070, 0.06780, 0.06590, 0.06520, 0.06400, 0.06280, 0.06220, 0.06080, 0.05780])
mu_c = np.array([0.06950, 0.06650, 0.06430, 0.06350, 0.06200, 0.06060, 0.06000, 0.05850, 0.05550])
mu_p = np.array([0.00040, 0.00030, 0.00025, 0.00022, 0.00020, 0.00017, 0.00015, 0.00013, 0.00010])
mu_pp = np.array([0.0, 0.0, 0.00020, 0.00028, 0.00050, 0.00080, 0.00100, 0.00130, 0.00150])
mu_r = np.array([0.00080, 0.00068, 0.00060, 0.00055, 0.00050, 0.00043, 0.00040, 0.00035, 0.00030])

f_t = interp1d(e_arr, mu_t, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_c = interp1d(e_arr, mu_c, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_p = interp1d(e_arr, mu_p, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_pp = interp1d(e_arr, mu_pp, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_r = interp1d(e_arr, mu_r, kind='cubic', bounds_error=False, fill_value='extrapolate')

def mu_tot(E): return float(f_t(np.clip(E, 0.01, 10.0)))
def mu_compton(E): return float(f_c(np.clip(E, 0.01, 10.0)))
def mu_photoel(E): return float(f_p(np.clip(E, 0.01, 10.0)))
def mu_pairprod(E): return float(f_pp(np.clip(E, 0.01, 10.0)))
def mu_rayleigh(E): return float(f_r(np.clip(E, 0.01, 10.0)))

# ============================================================
# Geometry
# ============================================================
SSD = 100.0
field_half = 5.0
z0 = 0.0
zc = 5.0
zh = 0.1
z_low = zc - zh
z_high = zc + zh
cyl_r = 1.0
src_pos = np.array([0.0, 0.0, -SSD])

# Pyramid solid angle
th_half = np.arctan(field_half / SSD)
Omega_pyr = 4 * np.arcsin(np.sin(th_half)**2)

# Electron density
n_e = 10 * N_A / 18.015

print(f"Omega_pyr = {Omega_pyr:.6f} sr")
print(f"n_e = {n_e:.4e} e-/cm^3")

# ============================================================
# Helper functions
# ============================================================
def compton_Eout(E_in, costh):
    return E_in / (1 + (E_in / m_e) * (1 - costh))

def kn_dcs(E_in, costh):
    th = np.arccos(np.clip(costh, -1, 1))
    E_out = compton_Eout(E_in, costh)
    return (r_e**2 / 2) * (E_out/E_in)**2 * (E_in/E_out + E_out/E_in - np.sin(th)**2)

def sample_compton(E_in):
    max_d = r_e**2 * 1.1
    while True:
        ct = np.random.uniform(-1, 1)
        d = kn_dcs(E_in, ct)
        if np.random.random() * max_d < d:
            break
    E_out = compton_Eout(E_in, ct)
    t_e = E_in - E_out
    num = E_in - E_out * ct
    den = np.sqrt(E_in**2 + E_out**2 - 2*E_in*E_out*ct)
    ce = num / den if den > 0 else 1.0
    return E_out, ct, t_e, ce

def rotate_vec(v, costh):
    u = v / np.linalg.norm(v)
    th = np.arccos(np.clip(costh, -1, 1))
    ph = np.random.uniform(0, 2*np.pi)
    if abs(u[2]) < 0.999:
        w = np.cross(u, np.array([0,0,1]))
    else:
        w = np.cross(u, np.array([1,0,0]))
    w = w / np.linalg.norm(w)
    v2 = np.cross(w, u)
    return u * costh + w * np.sin(th) * np.cos(ph) + v2 * np.sin(th) * np.sin(ph)

# Hmm, I have errors. Let me rewrite more carefully.