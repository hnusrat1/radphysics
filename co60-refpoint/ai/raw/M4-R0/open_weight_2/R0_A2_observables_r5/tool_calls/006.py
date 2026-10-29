import numpy as np

with open('/tmp/data/estar_water.txt', 'r') as f:
    estar_lines = f.readlines()

estar_data = []
for line in estar_lines[4:]:
    parts = line.split()
    if len(parts) >= 7:
        try:
            energy = float(parts[0])
            collision_sp = float(parts[1])
            radiative_sp = float(parts[2])
            total_sp = float(parts[3])
            csda_range = float(parts[4])
            radiation_yield = float(parts[5])
            density_effect = float(parts[6])
            # No 8th column (delta_ray_cutoff)
            estar_data.append([energy, collision_sp, radiative_sp, total_sp, 
                             csda_range, radiation_yield, density_effect])
        except:
            pass

estar_array = np.array(estar_data)
print(f"ESTAR entries: {len(estar_array)}")
print(f"Energy range: {estar_array[0,0]:.6f} to {estar_array[-1,0]:.6f} MeV")
print(f"\nFirst row: {estar_array[0]}")
print(f"Last row: {estar_array[-1]}")

# Interpolate function
def interpolate_estar_log(E):
    energies = estar_array[:, 0]
    idx = np.searchsorted(energies, E)
    if idx == 0 or idx >= len(energies):
        return None
    
    log_E = np.log10(max(E, 1e-20))
    log_E_low = np.log10(energies[idx-1])
    log_E_high = np.log10(energies[idx])
    
    t = (log_E - log_E_low) / (log_E_high - log_E_low)
    
    result = {}
    cols_names = ['collision', 'radiative', 'total', 'csda']
    col_indices = [1, 2, 3, 4]
    
    for name, idx_col in zip(cols_names, col_indices):
        val_low = estar_array[idx-1, idx_col]
        val_high = estar_array[idx, idx_col]
        
        if val_low > 0 and val_high > 0:
            result[name] = val_low * (val_high/val_low)**t
        elif val_high > 0:
            result[name] = val_high
        else:
            result[name] = 0.0
    
    result['density_effect'] = estar_array[idx, 6]
    return result

# Test at some relevant energies
for E_test in [0.2, 0.5, 0.8, 1.0, 1.2, 1.3, 1.5, 2.0]:
    est = interpolate_estar_log(E_test)
    if est:
        ratio = est['radiative']/est['collision'] if est['collision'] > 0 else 0
        print(f"{E_test:5.2f} MeV: Coll={est['collision']:8.4f}, Rad={est['radiative']:9.6f}, Ratio={ratio:.4f}")