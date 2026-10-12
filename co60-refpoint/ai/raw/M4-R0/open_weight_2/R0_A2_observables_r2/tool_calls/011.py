import numpy as np

# Load data
xcom = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
estar = np.loadtxt('/tmp/data/estar_water.txt', comments='#')

# Photon energies and weights
E_photons = np.array([1.1732, 1.3325])  # MeV
w_photons = np.array([0.5, 0.5])

def interp_xcom_column(col_idx, E):
    energies = xcom[:, 0]
    values = xcom[:, col_idx]
    return np.interp(np.log10(E), np.log10(energies), values)

def interp_estar_column(col_idx, E):
    energies = estar[:, 0]
    values = estar[:, col_idx]
    return np.interp(E, energies, values)

# Mass attenuation coefficients (cm^2/g) for weighted average
mu_coherent_weighted = sum(w * interp_xcom_column(1, E) for w, E in zip(w_photons, E_photons))
mu_compton_weighted = sum(w * interp_xcom_column(2, E) for w, E in zip(w_photons, E_photons))
mu_photo_weighted = sum(w * interp_xcom_column(3, E) for w, E in zip(w_photons, E_photons))
mu_pair_nuc_weighted = sum(w * interp_xcom_column(4, E) for w, E in zip(w_photons, E_photons))
mu_pair_ele_weighted = sum(w * interp_xcom_column(5, E) for w, E in zip(w_photons, E_photons))
mu_total_with_coh_weighted = sum(w * interp_xcom_column(6, E) for w, E in zip(w_photons, E_photons))
mu_total_no_coh_weighted = sum(w * interp_xcom_column(7, E) for w, E in zip(w_photons, E_photons))

# Linear attenuation coefficients (cm^-1), assuming density = 1.0 g/cm^3
rho = 1.0  # g/cm^3
mu_lin_weighted = mu_total_with_coh_weighted * rho

print("Weighted mass attenuation coefficients (cm^2/g):")
print(f"  Total with coherent: {mu_total_with_coh_weighted:.6e}")
print(f"  Total without coherent: {mu_total_no_coh_weighted:.6e}")
print(f"  Compton: {mu_compton_weighted:.6e}")
print(f"  Coherent: {mu_coherent_weighted:.6e}")
print(f"  Photoelectric: {mu_photo_weighted:.6e}")
print(f"  Pair nuclear: {mu_pair_nuc_weighted:.6e}")
print(f"\nLinear attenuation coefficient (cm^-1): {mu_lin_weighted:.6e}")

# Fraction of primary photons transmitted through 5 cm of water
depth = 5.0  # cm
primary_transmission = np.exp(-mu_lin_weighted * depth)
print(f"\nPrimary transmission fraction through 5 cm: {primary_transmission:.6f}")

# This is L1_primary_fluence_per_phi0
L1_primary = primary_transmission
print(f"L1_primary_fluence_per_phi0 = {L1_primary:.6f}")
