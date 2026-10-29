import numpy as np

# Read estar_water.txt
data = []
with open('/tmp/data/estar_water.txt', 'r') as f:
    for line in f:
        if not line.startswith('#'):
            parts = [float(x) for x in line.split()]
            data.append(parts)

estar = np.array(data)
print("ESTAR shape:", estar.shape)
print("Columns: electron kinetic energy (MeV), collision stopping power, radiative stopping power, total stopping power, CSDA range, radiation yield, density-effect parameter")
print("\nFirst row:", estar[0])
print("\nLast row:", estar[-1])
print("\nElectron energy range:", estar[:, 0].min(), "to", estar[:, 0].max(), "MeV")

# Check where our energies of interest fall
for e in [0.01, 0.511, 1.0, 2.0]:
    idx = np.argmin(np.abs(estar[:, 0] - e))
    print(f"\nClosest to {e} MeV: index {idx}, energy = {estar[idx, 0]} MeV")
    print(f"  Collision: {estar[idx, 1]:.6e}, Radiative: {estar[idx, 2]:.6e}, Total: {estar[idx, 3]:.6e}")