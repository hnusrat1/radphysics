import numpy as np

# Reload data with proper parsing
with open('/tmp/data/xcom_water.txt', 'r') as f:
    xcom_lines = f.readlines()
with open('/tmp/data/estar_water.txt', 'r') as f:
    estar_lines = f.readlines()

xcom_data = []
for line in xcom_lines[4:]:  # Skip headers (first 4-5 lines)
    if line.startswith('#'):
        continue
    parts = line.split()
    if len(parts) >= 7:
        try:
            vals = [float(p) for p in parts]
            xcom_data.append(vals)
        except:
            pass
xcom_array = np.array(xcom_data)

estar_data = []
for line in estar_lines[4:]:
    if line.startswith('#'):
        continue
    parts = line.split()
    if len(parts) >= 7:
        try:
            vals = [float(p) for p in parts]
            estar_data.append(vals)
        except:
            pass
estar_array = np.array(estar_data)

print(f"XCOM entries: {len(xcom_array)}, shape: {xcom_array.shape}")
print(f"ESTAR entries: {len(estar_array)}, shape: {estar_array.shape}")

def interpolate_xlog(E, col_idx):
    energies = xcom_array[:, 0]
    idx = np.searchsorted(energies, E)
    if idx == 0 or idx >= len(energies):
        return None
    log_E_low = np.log10(max(energies[idx-1], 1e-20))
    log_E_high = np.log10(max(energies[idx], 1e-20))
    t = (np.log10(max(E, 1e-20)) - log_E_low) / (log_E_high - log_E_low)
    val_low, val_high = xcom_array[idx-1, col_idx], xcom_array[idx, col_idx]
    if val_low > 0 and val_high > 0:
        return val_low * (val_high/val_low)**t
    elif val_high > 0:
        return val_high
    return 0.0

def interpolate_estar_log(E):
    energies = estar_array[:, 0]
    idx = np.searchsorted(energies, E)
    if idx == 0 or idx >= len(energies):
        return None
    log_E_low = np.log10(max(energies[idx-1], 1e-20))
    log_E_high = np.log10(max(energies[idx], 1e-20))
    t = (np.log10(max(E, 1e-20)) - log_E_low) / (log_E_high - log_E_low)
    result = {}
    for name, idx_col in [('collision',1), ('radiative',2), ('total',3)]:
        val_low, val_high = estar_array[idx-1, idx_col], estar_array[idx, idx_col]
        if val_low > 0 and val_high > 0:
            result[name] = val_low * (val_high/val_low)**t
        else:
            result[name] = val_high if val_high > 0 else 0.0
    return result

E_co1, E_co2 = 1.1732, 1.3325
rho = 1.0
depth_cm = 5.0

# Cross sections
xs = {}
for E in [E_co1, E_co2]:
    xs[E] = {k: interpolate_xlog(E, i+1) for i, k in enumerate(['coherent','incoherent','photoelectric','pair_nuclear','pair_electron','total'])}

print("\nCross sections at Co-60 energies:")
for E, label in [(E_co1, "E1"), (E_co2, "E2")]:
    print(f"\n{label} ({E} MeV):")
    for k, v in xs[E].items():
        print(f"  {k}: {v:.6e}")

# Interaction fractions
frac_types = {}
for key in ['compton', 'photoelectric', 'pair', 'rayleigh']:
    if key == 'compton':
        key_names = ['incoherent']
    elif key == 'photoelectric':
        key_names = ['photoelectric']
    elif key == 'pair':
        key_names = ['pair_nuclear', 'pair_electron']
    else:  # rayleigh
        key_names = ['coherent']
    
    frac_val = 0.0
    for kn in key_names:
        frac_val += 0.5 * xs[E_co1][kn]/xs[E_co1]['total'] + 0.5 * xs[E_co2][kn]/xs[E_co2]['total']
    frac_types[key] = frac_val

print("\nInteraction fractions (averaged over two lines):")
for k,v in frac_types.items():
    print(f"  {k}: {v:.6f}")
print(f"  Sum: {sum(frac_types.values()):.6f}")

# Mean Compton energy transfer fraction
def mean_compton_frac(E):
    costheta = np.linspace(-1, 1, 1000)
    alpha = E/0.511
    sigma_ws = []
    T_over_E_ws = []
    for ct in costheta:
        theta = np.arccos(np.clip(ct, -1, 1))
        E_prime = E / (1 + alpha*(1-ct))
        T = E - E_prime
        r0sq = 0.066524e-24/(4*np.pi)
        factor = (1+ct**2)/2
        epe = E_prime/E
        sig = factor * epe**2 * r0sq
        sigma_ws.append(sig)
        T_over_E_ws.append(T/E)
    sigma_ws = np.array(sigma_ws)
    T_over_E_ws = np.array(T_over_E_ws)
    return np.sum(T_over_E_ws*sigma_ws)/np.sum(sigma_ws)

mean_T_over_E_co1 = mean_compton_frac(E_co1)
mean_T_over_E_co2 = mean_compton_frac(E_co2)
print(f"\nMean Compton energy transfer fractions:")
print(f"  E={E_co1}: {mean_T_over_E_co1:.4f}")
print(f"  E={E_co2}: {mean_T_over_E_co2:.4f}")