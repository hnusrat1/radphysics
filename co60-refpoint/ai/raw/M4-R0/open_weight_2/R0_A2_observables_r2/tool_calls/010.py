import numpy as np

# Load data
xcom = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
estar = np.loadtxt('/tmp/data/estar_water.txt', comments='#')

# Define photon energies (Co-60 lines)
E_photons = np.array([1.1732, 1.3325])  # MeV
E_fraction = np.array([0.5, 0.5])  # Equal probability

# Define bins
photon_energy_bins = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
electron_energy_bins = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.4])
cos_theta_bins = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

def interp_xcom_column(col_idx, E):
    """Interpolate XCOM column value at given energy E (MeV)"""
    energies = xcom[:, 0]
    values = xcom[:, col_idx]
    # Use log-log interpolation as stated in the file header
    return np.interp(np.log10(E), np.log10(energies), values)

def interp_estar_column(col_idx, E):
    """Interpolate ESTAR column value at given energy E (MeV)"""
    energies = estar[:, 0]
    values = estar[:, col_idx]
    return np.interp(E, energies, values)

# Compute attenuation coefficients for Co-60 energies
print("Cross sections at Co-60 energies:")
for i, E in enumerate(E_photons):
    mu_coherent = interp_xcom_column(1, E)  # Col 1: coherent
    mu_compton = interp_xcom_column(2, E)   # Col 2: incoherent
    mu_photo = interp_xcom_column(3, E)      # Col 3: photoelectric
    mu_pair_nuc = interp_xcom_column(4, E)   # Col 4: pair nuclear
    mu_pair_ele = interp_xcom_column(5, E)   # Col 5: pair electron
    mu_total_coh = interp_xcom_column(6, E)  # Col 6: total with coherent
    mu_total_no_coh = interp_xcom_column(7, E)  # Col 7: total without coherent
    
    print(f"\nE = {E:.4f} MeV:")
    print(f"  Coherent: {mu_coherent:.6e} cm^2/g")
    print(f"  Compton: {mu_compton:.6e} cm^2/g")
    print(f"  Photoelectric: {mu_photo:.6e} cm^2/g")
    print(f"  Pair (nuc): {mu_pair_nuc:.6e}")
    print(f"  Pair (ele): {mu_pair_ele:.6e}")
    print(f"  Total (with coh): {mu_total_coh:.6e}")
    print(f"  Total (no coh): {mu_total_no_coh:.6e}")
