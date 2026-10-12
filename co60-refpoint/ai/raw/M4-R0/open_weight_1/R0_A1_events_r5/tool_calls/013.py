import numpy as np
from scipy.interpolate import interp1d

np.random.seed(42)

# Test primary rate with debugging
m_e = 0.51099895
r_e = 2.8179403262e-13
NA = 6.02214076e23

e_arr = np.array([1.0, 1.1, 1.1732, 1.2, 1.25, 1.3, 1.3325, 1.4, 1.5])
mu_tot_v = np.array([0.07070, 0.06780, 0.06590, 0.06520, 0.06400, 0.06280, 0.06220, 0.06080, 0.05780])
mu_comp_v = np.array([0.06950, 0.06650, 0.06430, 0.06350, 0.06200, 0.06060, 0.06000, 0.05850, 0.05550])
mu_phot_v = np.array([0.00040, 0.00030, 0.00025, 0.00022, 0.00020, 0.00017, 0.00015, 0.00013, 0.00010])
mu_pair_v = np.array([0.0, 0.0, 0.00020, 0.00028, 0.00050, 0.00080, 0.00100, 0.00130, 0.00150])
mu_rayl_v = np.array([0.00080, 0.00068, 0.00060, 0.00055, 0.00050, 0.00043, 0.00040, 0.00035, 0.00030])

f_tot = interp1d(e_arr, mu_tot_v, kind='cubic', bounds_error=False, fill_value='extrapolate')

def mu_t(E): return float(f_tot(np.clip(E, 0.01, 10.0)))

SSD = 100.0
src = np.array([0.0, 0.0, -SSD])
field_half = 5.0
z_surf = 0.0
z_ctr = 5.0
z_half = 0.1
z_lo = z_ctr - z_half
z_hi = z_ctr + z_half
cyl_R = 1.0

th_half = np.arctan(field_half / SSD)
Omega_pyr = 4 * np.arcsin(np.sin(th_half)**2)

# Debug: trace 10000 samples
N_test = 100000
hits = 0
total_w = 0.0

for _ in range(N_test):
    E = 1.1732 if np.random.random() < 0.5 else 1.3325
    x = np.random.uniform(-field_half, field_half)
    y = np.random.uniform(-field_half, field_half)
    d = np.array([x - src[0], y - src[1], z_surf - src[2]])
    d = d / np.linalg.norm(d)
    dz = d[2]
    
    if dz <= 0:
        continue
    
    t_sc = (z_lo - z_surf) / dz
    ep = np.array([0.0, 0.0, z_surf]) + t_sc * d
    r_ep = np.sqrt(ep[0]**2 + ep[1]**2)
    
    if r_ep <= cyl_R:
        hits += 1
        path = 0.2 / dz
        muE = mu_t(E)
        att = np.exp(-muE * t_sc)
        p_int = 1.0 - np.exp(-muE * path)
        total_w += att * p_int

avg_w = total_w / N_test
print(f"Total samples: {N_test}")
print(f"Hits: {hits} ({100*hits/N_test:.2f}%)")
print(f"Sum of weights: {total_w:.6f}")
print(f"Average weight: {avg_w:.8f}")
print(f"Expected average: ~{0.73*0.0128*0.0285:.8f}")
print(f"Per hit weight: {total_w/hits if hits > 0 else 0:.6f}")