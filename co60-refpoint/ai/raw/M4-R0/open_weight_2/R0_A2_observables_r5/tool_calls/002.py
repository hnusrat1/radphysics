import numpy as np

# Parse XCOM data
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
            total_without_coherent = float(parts[6])  # same column
            xcom_data.append([energy, coherent, incoherent, photoelectric, 
                             pair_nuclear, pair_electron, total_with_coherent])
        except:
            pass

xcom_array = np.array(xcom_data)
print("XCOM columns:")
print("Energy (MeV), Coherent, Incoherent, Photoelectric, Pair_nuc, Pair_el, Total")
print(f"Energy range: {xcom_array[0,0]:.6f} to {xcom_array[-1,0]:.6f} MeV")

# Interpolate for our energies
E1 = 1.1732
E2 = 1.3325

def interpolate_xcom(E):
    """Interpolate XCOM cross sections logarithmically"""
    energies = xcom_array[:, 0]
    
    # Find surrounding points
    idx = np.searchsorted(energies, E)
    if idx == 0 or idx >= len(energies):
        return None
    
    # Log-log interpolation
    log_E = np.log10(E)
    log_E_low = np.log10(energies[idx-1])
    log_E_high = np.log10(energies[idx])
    
    t = (log_E - log_E_low) / (log_E_high - log_E_low)
    
    result = {}
    for i, col in enumerate(['coherent', 'incoherent', 'photoelectric', 
                            'pair_nuclear', 'pair_electron', 'total']):
        val_low = xcom_array[idx-1, i+1]
        val_high = xcom_array[idx, i+1]
        result[col] = val_low * (val_high/val_low)**t if val_low > 0 and val_high > 0 else 0
    
    return result

for E in [E1, E2]:
    xs = interpolate_xcom(E)
    print(f"\n{E} MeV:")
    print(f"  Coherent (Rayleigh): {xs['coherent']:.6e} cm^2/g")
    print(f"  Incoherent (Compton): {xs['incoherent']:.6e} cm^2/g")
    print(f"  Photoelectric: {xs['photoelectric']:.6e} cm^2/g")
    print(f"  Pair nuclear: {xs['pair_nuclear']:.6e} cm^2/g")
    print(f"  Pair electron: {xs['pair_electron']:.6e} cm^2/g")
    print(f"  Total: {xs['total']:.6e} cm^2/g")