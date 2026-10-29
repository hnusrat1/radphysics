import numpy as np

# Re-read XCOM data
with open('/tmp/data/xcom_water.txt', 'r') as f:
    xcom_lines = f.readlines()

xcom_data = []
for line in xcom_lines[4:]:  # Skip header lines (first 4 are comments)
    parts = line.split()
    if len(parts) >= 7:
        try:
            energy = float(parts[0])
            coherent = float(parts[1])
            incoherent = float(parts[2])
            photoelectric = float(parts[3])
            pair_nuclear = float(parts[4])
            pair_electron = float(parts[5])
            total_with_coherent = float(parts[6])
            xcom_data.append([energy, coherent, incoherent, photoelectric, 
                             pair_nuclear, pair_electron, total_with_coherent])
        except:
            pass

xcom_array = np.array(xcom_data)
print(f"XCOM entries: {len(xcom_array)}")
print(f"Energy range: {xcom_array[0,0]:.6f} to {xcom_array[-1,0]:.6f} MeV")

# Interpolate function for log-log
def interpolate_xlog(E, col_idx):
    energies = xcom_array[:, 0]
    idx = np.searchsorted(energies, E)
    if idx == 0 or idx >= len(energies):
        return None
    
    log_E = np.log10(max(E, 1e-20))
    log_E_low = np.log10(energies[idx-1])
    log_E_high = np.log10(energies[idx])
    
    t = (log_E - log_E_low) / (log_E_high - log_E_low)
    
    val_low = xcom_array[idx-1, col_idx]
    val_high = xcom_array[idx, col_idx]
    
    if val_low > 0 and val_high > 0:
        return val_low * (val_high/val_low)**t
    elif val_high > 0:
        return val_high
    else:
        return 0.0

E1 = 1.1732
E2 = 1.3325

cols = ['coherent', 'incoherent', 'photoelectric', 'pair_nuclear', 'pair_electron', 'total']
col_indices = [1, 2, 3, 4, 5, 6]

for E in [E1, E2]:
    print(f"\n{E} MeV:")
    for name, idx in zip(cols, col_indices):
        val = interpolate_xlog(E, idx)
        print(f"  {name}: {val:.6e} cm^2/g")