import numpy as np
import json

# Load data
xcom = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
estar = np.loadtxt('/tmp/data/estar_water.txt', comments='#')

def interp_xcom(ci, E):
    return np.interp(np.log10(E), np.log10(xcom[:, 0]), xcom[:, ci])

def interp_estar(ci, E):
    return np.interp(E, estar[:, 0], estar[:, ci])

E_co = np.array([1.1732, 1.3325])
w_co = np.array([0.5, 0.5])
depth = 5.0
rho = 1.0

pe_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                     0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
                     1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
ee_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                     0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                     1, 1.05, 1.1, 1.15, 1.2, 1.4])
ct_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

# Layer 1 calculation
mu_t_w = sum(w * interp_xcom(6, E) for w, E in zip(w_co, E_co))
mu_l = mu_t_w * rho
primary_frac = np.exp(-mu_l * depth)
b_sp = 0.18
total_frac = primary_frac * (1 + b_sp)
scatter_frac_fluence = 1 - primary_frac / total_frac

mean_E_primary = np.average(E_co, weights=w_co)
mean_E_scattered = mean_E_primary * 0.75
efluo_primary = primary_frac * mean_E_primary
efluo_scattered = (total_frac - primary_frac) * mean_E_scattered
scatter_frac_efluo = eflua_secondary if 'eflua_secondary' not in globals() else eflua_secondary
mean_E_all = (eflua_primary + efluoa_secondary) / total_frac

print(f"L1_primary={primary_frac:.6f}, L1_total={total_frac:.6f}")
print(f"Mean E primary={mean_E_primary:.3f}, Mean E scattered={mean_E_scattered:.3f}")
