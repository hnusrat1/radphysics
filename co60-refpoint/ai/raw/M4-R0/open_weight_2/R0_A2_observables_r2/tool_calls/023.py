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

# Calculate layer 1
mu_t_w = sum(w * interp_xcom(6, E) for w, E in zip(w_co, E_co))
mu_l = mu_t_w * rho
primary_frac_val = np.exp(-mu_l * depth)
b_sp = 0.18
total_frac_val = primary_frac_val * (1 + b_sp)
scatter_frac_fluence_val = 1 - primary_frac_val / total_frac_val

mean_E_primary = np.average(E_co, weights=w_co)
mean_E_scattered = mean_E_primary * 0.75
flu_o_pri = primary_frac_val * mean_E_primary
flu_o_sec = (total_frac_val - primary_frac_val) * mean_E_scattered
scatter_frac_efluo_val = fluoa_second / (flua_first + fluoa_secondary) if False else fluoa_second / (flua_primary + eflua_secondary)
mean_E_all_val = (eflua_first + efual_second) / total_frac_val

print(f"L1 primary={primary_frac_val:.4f}, total={total_frac_val:.4f}")
