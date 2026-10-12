import numpy as np

# Load XCOM data
xcom_data = np.loadtxt('/tmp/data/xcom_water.txt')
xcom_energies = xcom_data[:, 0]
xcom_coherent = xcom_data[:, 1]
xcom_incoherent = xcom_data[:, 2]
xcom_photo = xcom_data[:, 3]
xcom_pair_nuc = xcom_data[:, 4]
xcom_pair_elec = xcom_data[:, 5]
xcom_total_with = xcom_data[:, 6]
xcom_total_without = xcom_data[:, 7]

# At 2.0 MeV
idx = np.searchsorted(xcom_energies, 2.0)
print(f"idx={idx}, E={xcom_energies[idx]:.6f}")
print(f"coherent={xcom_coherent[idx]:.6e}")
print(f"incoherent={xcom_incoherent[idx]:.6e}")
print(f"photo={xcom_photo[idx]:.6e}")
print(f"pair_nuc={xcom_pair_nuc[idx]:.6e}")
print(f"pair_elec={xcom_pair_elec[idx]:.6e}")
print(f"total_with={xcom_total_with[idx]:.6e}")

# Interpolation function
def interp_xcom(E, y):
    return np.interp(E, xcom_energies, y)

# Get values at exactly 2.0 MeV
mu_coherent = interp_xcom(2.0, xcom_coherent)
mu_incoherent = interp_xcom(2.0, xcom_incoherent)
mu_photo = interp_xcom(2.0, xcom_photo)
mu_pair_nuc = interp_xcom(2.0, xcom_pair_nuc)
mu_pair_elec = interp_xcom(2.0, xcom_pair_elec)
mu_total = interp_xcom(2.0, xcom_total_with)

print(f"\nAt 2.0 MeV (interpolated):")
print(f"mu_coherent = {mu_coherent:.6e}")
print(f"mu_incoherent = {mu_incoherent:.6e}")
print(f"mu_photo = {mu_photo:.6e}")
print(f"mu_pair_nuc = {mu_pair_nuc:.6e}")
print(f"mu_pair_elec = {mu_pair_elec:.6e}")
print(f"mu_total = {mu_total:.6e}")

# Check pair electron field - the data says 0 for 2 MeV, but pair production in electron field
# should be non-zero above 1.022 MeV. Let me check.
print(f"\nPair electron at various energies:")
for e in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    idx = np.searchsorted(xcom_energies, e)
    print(f"  E={e}: pair_elec={xcom_pair_elec[idx]:.6e}, pair_nuc={xcom_pair_nuc[idx]:.6e}")