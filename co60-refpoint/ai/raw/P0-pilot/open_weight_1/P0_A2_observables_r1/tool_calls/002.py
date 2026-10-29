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

print("XCOM energies (first 10):", xcom_energies[:10])
print("XCOM energies (last 10):", xcom_energies[-10:])

# Load ESTAR data
estar_data = np.loadtxt('/tmp/data/estar_water.txt')
estar_energies = estar_data[:, 0]
estar_coll = estar_data[:, 1]
estar_rad = estar_data[:, 2]
estar_total = estar_data[:, 3]
estar_csda = estar_data[:, 4]
estar_rad_yield = estar_data[:, 5]
estar_delta = estar_data[:, 6]

print("\nESTAR energies (first 10):", estar_energies[:10])
print("ESTAR energies (last 10):", estar_energies[-10:])

# Check around 2.0 MeV
print("\nXCOM at ~2.0 MeV:")
idx = np.searchsorted(xcom_energies, 2.0)
print(f"idx={idx}, E={xcom_energies[idx]:.6f}, coherent={xcom_coherent[idx]:.6e}, incoherent={xcom_incoherent[idx]:.6e}, photo={xcom_photo[idx]:.6e}, pair_nuc={xcom_pair_nuc[idx]:.6e}, pair_elec={xcom_pair_elec[idx]:.6e}, total={xcom_total_with[idx]:.6e}")
if idx > 0:
    print(f"idx-1: E={xcom_energies[idx-1]:.6f}, coherent={xcom_coherent[idx-1]:.6e}, incoherent={xcom_incoherent[idx-1]:.6e}, photo={xcom_photoe[idx-1]:.6e}, pair_nuc={xcom_pair_nuc[idx-1]:.6e}, pair_elec={xcom_pair_elec[idx-1]:.6e}, total={xcom_total_with[idx-1]:.6e}")

print("\nESTAR at ~2.0 MeV:")
idx = np.searchsorted(estar_energies, 2.0)
print(f"idx={idx}, E={estar_energies[idx]:.6f}, coll={estar_coll[idx]:.6f}, rad={estar_rad[idx]:.6f}, csda={estar_csda[idx]:.6f}")