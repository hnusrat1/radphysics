import numpy as np
from scipy.interpolate import interp1d

# Re-read all data
with open('/tmp/data/xcom_water.txt', 'r') as f:
    xcom_lines = f.readlines()
    
with open('/tmp/data/estar_water.txt', 'r') as f:
    estar_lines = f.readlines()

# Parse XCOM
xcom_data = []
for line in xcom_lines[4:]:
    parts = line.split()
    if len(parts) >= 7:
        try:
            vals = [float(p) for p in parts]
            xcom_data.append(vals)
        except:
            pass
xcom_array = np.array(xcom_data)

# Parse ESTAR
estar_data = []
for line in estar_lines[4:]:
    parts = line.split()
    if len(parts) >= 7:
        try:
            vals = [float(p) for p in parts]
            estar_data.append(vals)
        except:
            pass
estar_array = np.array(estar_data)

def interpolate_xlog(E, col_idx):
    """Log-log interpolation of XCOM data"""
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

def interpolate_estar_log(E):
    """Log-log interpolation of ESTAR data"""
    energies = estar_array[:, 0]
    idx = np.searchsorted(energies, E)
    if idx == 0 or idx >= len(energies):
        return None
    
    log_E = np.log10(max(E, 1e-20))
    log_E_low = np.log10(energies[idx-1])
    log_E_high = np.log10(energies[idx])
    
    t = (log_E - log_E_low) / (log_E_high - log_E_low)
    
    result = {}
    cols_names = ['collision', 'radiative', 'total']
    col_indices = [1, 2, 3]
    
    for name, idx_col in zip(cols_names, col_indices):
        val_low = estar_array[idx-1, idx_col]
        val_high = estar_array[idx, idx_col]
        
        if val_low > 0 and val_high > 0:
            result[name] = val_low * (val_high/val_low)**t
        elif val_high > 0:
            result[name] = val_high
        else:
            result[name] = 0.0
    
    return result

# Constants
E_co1 = 1.1732  # MeV
E_co2 = 1.3325  # MeV
density_water = 1.0  # g/cm^3
rho = density_water
mu_over_rho = {}  # mass attenuation coefficient cm^2/g

# Get cross sections for Co-60 lines
for E, name in [(E_co1, 'co1'), (E_co2, 'co2')]:
    coherent = interpolate_xlog(E, 1)
    incoherent = interpolate_xlog(E, 2)
    photoelectric = interpolate_xlog(E, 3)
    pair_nuc = interpolate_xlog(E, 4)
    pair_el = interpolate_xlog(E, 5)
    total = interpolate_xlog(E, 6)
    
    mu_over_rho[name] = {
        'coherent': coherent,
        'incoherent': incoherent,
        'photoelectric': photoelectric,
        'pair_nuclear': pair_nuc,
        'pair_electron': pair_el,
        'total': total
    }
    
print("Cross sections (cm^2/g):")
for name, xs in mu_over_rho.items():
    print(f"{name}:")
    for k, v in xs.items():
        print(f"  {k}: {v:.6e}")

# Calculate attenuation factor through 5 cm water
# For primary photons
def attenuation(E, thickness_cm):
    xs = mu_over_rho['co1'] if E == E_co1 else mu_over_rho['co2']
    mu = xs['total'] * rho  # cm^-1
    return np.exp(-mu * thickness_cm)

att_5cm_co1 = attenuation(E_co1, 5.0)
att_5cm_co2 = attenuation(E_co2, 5.0)
print(f"\nAttenuation through 5 cm:")
print(f"  {E_co1} MeV: {att_5cm_co1:.6f}")
print(f"  {E_co2} MeV: {att_5cm_co2:.6f}")

# Average attenuation (equal probability)
avg_att = 0.5 * att_5cm_co1 + 0.5 * att_5cm_co2
print(f"  Average: {avg_att:.6f}")