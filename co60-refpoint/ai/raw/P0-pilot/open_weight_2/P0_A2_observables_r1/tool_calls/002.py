import numpy as np

# Read xcom_water.txt
data = []
with open('/tmp/data/xcom_water.txt', 'r') as f:
    for line in f:
        if not line.startswith('#'):
            parts = [float(x) for x in line.split()]
            data.append(parts)

xcom = np.array(data)
print("XCOM shape:", xcom.shape)
print("Columns: photon energy (MeV), coherent, incoherent, photoelectric, pair_nuclear, pair_electron, total_with_coherent, total_without_coherent")
print("\nFirst row:", xcom[0])
print("\nLast row:", xcom[-1])
print("\nPhoton energy range:", xcom[:, 0].min(), "to", xcom[:, 0].max(), "MeV")

# Check where our 2.0 MeV photon energy falls
energy_idx = np.argmin(np.abs(xcom[:, 0] - 2.0))
print(f"\nClosest energy bin to 2.0 MeV: index {energy_idx}, energy = {xcom[energy_idx, 0]} MeV")