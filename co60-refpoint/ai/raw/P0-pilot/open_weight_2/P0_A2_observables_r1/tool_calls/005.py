import numpy as np

# Read xcom_data again
xcom_data = []
with open('/tmp/data/xcom_water.txt', 'r') as f:
    for line in f:
        if not line.startswith('#'):
            parts = [float(x) for x in line.split()]
            xcom_data.append(parts)
xcom = np.array(xcom_data)

print("Energy column values:")
for i, e in enumerate(xcom[:, 0]):
    if 1.5 <= e <= 2.5:
        print(f"Index {i}: E={e:.4f} MeV, Rayleigh={xcom[i, 1]:.6e}, Compton={xcom[i, 2]:.6e}, Photo={xcom[i, 3]:.6e}, Pair_nuc={xcom[i, 4]:.6e}, Pair_el={xcom[i, 5]:.6e}")

# Check exact value at 2.0 MeV
idx_2mev = np.argmin(np.abs(xcom[:, 0] - 2.0))
print(f"\nAt 2.0 MeV (index {idx_2mev}):")
print(f"  Energy: {xcom[idx_2mev, 0]:.6f}")
print(f"  Coherent: {xcom[idx_2mev, 1]:.6e}")
print(f"  Incoherent: {xcom[idx_2mev, 2]:.6e}")
print(f"  Photoelectric: {xcom[idx_2mev, 3]:.6e}")
print(f"  Pair nuclear: {xcom[idx_2mev, 4]:.6e}")
print(f"  Pair electron: {xcom[idx_2mev, 5]:.6e}")
print(f"  Total w/ coherent: {xcom[idx_2mev, 6]:.6e}")
print(f"  Total w/o coherent: {xcom[idx_2mev, 7]:.6e}")