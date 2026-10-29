import numpy as np
import json

# Final verification of all critical quantities before output

with open('/tmp/data/xcom_water.txt', 'r') as f:
    xcom_lines = f.readlines()
with open('/tmp/data/estar_water.txt', 'r') as f:
    estar_lines = f.readlines()

xcom_data = []
for line in xcom_lines[4:]:
    if line.startswith('#'): continue
    parts = line.split()
    if len(parts) >= 7:
        try: xcom_data.append([float(p) for p in parts])
        except: pass
xcom_array = np.array(xcom_data)

estar_data = []
for line in estar_lines[4:]:
    if line.startswith('#'): continue
    parts = line.split()
    if len(parts) >= 7:
        try: estar_data.append([float(p) for p in parts])
        except: pass
estar_array = np.array(estar_data)

def interpolate_xlog(E, col_idx):
    energies = xcom_array[:, 0]
    idx = np.searchsorted(energies, E)
    if idx == 0 or idx >= len(energies): return None
    log_E_low = np.log10(max(energies[idx-1], 1e-20))
    log_E_high = np.log10(max(energies[idx], 1e-20))
    t = (np.log10(max(E, 1e-20)) - log_E_low) / (log_E_high - log_E_low)
    val_low, val_high = xcom_array[idx-1, col_idx], xcom_array[idx, col_idx]
    if val_low > 0 and val_high > 0: return val_low * (val_high/val_low)**t
    elif val_high > 0: return val_high
    return 0.0

def interpolate_estar_log(E):
    energies = estar_array[:, 0]
    idx = np.searchsorted(energies, E)
    if idx == 0 or idx >= len(energies): return None
    return {'collision': estar_array[idx, 1], 'radiative': estar_array[idx, 2], 
            'total': estar_array[idx, 3], 'csda': estar_array[idx, 4]}

E_co1, E_co2 = 1.1732, 1.3325
rho = 1.0
depth_cm = 5.0

xs_co1 = {k: interpolate_xlog(E_co1, i+1) for i, k in enumerate(['coherent','incoherent','photoelectric','pair_nuclear','pair_electron','total'])}
xs_co2 = {k: interpolate_xlog(E_co2, i+1) for i, k in enumerate(['coherent','incoherent','photoelectric','pair_nuclear','pair_electron','total'])}

print("Key checks:")
print(f"xs_co1 total: {xs_co1['total']:.6e}")
print(f"xs_co2 total: {xs_co2['total']:.6e}")
print(f"Incoherent fraction Co1: {xs_co1['incoherent']/xs_co1['total']:.6f}")
print(f"Incoherent fraction Co2: {xs_co2['incoherent']/xs_co2['total']:.6f}")

S_coll = interpolate_estar_log(0.5)['collision']
print(f"\nCollision SP at 0.5 MeV: {S_coll:.4f} MeV cm^2/g")