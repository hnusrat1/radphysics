# Verify normalizations
import numpy as np

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

E_co1, E_co2 = 1.1732, 1.3325
xs_co1 = {k: interpolate_xlog(E_co1, i+1) for i, k in enumerate(['coherent','incoherent','photoelectric','pair_nuclear','pair_electron','total'])}

electron_energy_bins = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 
                        1.05, 1.1, 1.15, 1.2, 1.4]
n_el = len(electron_energy_bins)-1

ispec = np.zeros(n_el)
for Eg, w in [(E_co1, 0.5), (E_co2, 0.5)]:
    al = Eg/0.511; Epmin = Eg/(1+2*al); Tmax = Eg-Epmin
    for b in range(n_el):
        Em = (electron_energy_bins[b]+electron_energy_bins[b+1])/2
        if Em < Tmax:
            pdf = np.exp(-Em/(Tmax*0.3))
            ispec[b] += w*pdf*(electron_energy_bins[b+1]-electron_energy_bins[b])*xs_co1['incoherent']/xs_co1['total']

print(f"Initial spectrum sum: {ispec.sum():.6f}")
print(f"Bin edges: {electron_energy_bins[:5]}...{electron_energy_bins[-3:]}")
print(f"T_max for E={E_co1}: {1.1732-(1.1732/(1+2*1.1732/0.511)):.4f}")

# Check where bins end relative to T_max
for b in range(n_el):
    Em_high = electron_energy_bins[b+1]
    # For E1=1.17 MeV, T_max≈0.96 MeV
    # For E2=1.33 MeV, T_max≈1.12 MeV
    pass