# Write a complete script to a file
script = '''
import numpy as np
import json

xcom = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
estar = np.loadtxt('/tmp/data/estar_water.txt', comments='#')

def interp_x(ci, E):
    return np.interp(np.log10(E), np.log10(xcom[:, 0]), xcom[:, ci])

def interp_e(ci, E):
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

# Layer 1
mu_t_w = sum(w * interp_x(6, E) for w, E in zip(w_co, E_co))
mu_l = mu_t_w * rho
primary_val = np.exp(-mu_l * depth)
b_sp = 0.18
total_val = primary_val * (1 + b_sp)
scatter_frac_fluence = 1 - primary_val / total_val

mean_E_pri = np.average(E_co, weights=w_co)
mean_E_sc = mean_E_pri * 0.75
efluo_pri_val = primary_val * mean_E_pri
efluo_scat_val = (total_val - primary_val) * mean_E_sc
scatter_efluo_frac = eflua_sc / (flu_o_pri + efuala_sc) if False else eflua_sc / (eflua_pri + eflua_scat)
mean_all_E = (efluoa_pri + efual_sc) / total_val

print(f"Primary: {primary_val:.6f}")
print(f"Total: {total_val:.6f}")
print(f"Scatter frac fluence: {scatter_frac_fluence:.6f}")
print(f"Mean E all: {mean_all_E:.4f}")
'''

with open('/tmp/calc.py', 'w') as f:
    f.write(script)
    
exec(open('/tmp/calc.py').read())
