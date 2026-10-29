import numpy as np

xcom = np.loadtxt('/tmp/data/xcom_water.txt')
print("XCOM data shape:", xcom.shape)

# Look for energies near 1.17 and 1.33 MeV
for i in range(len(xcom)):
    e = xcom[i, 0]
    if 1.15 <= e <= 1.4:
        print(f"\nE={e:.4f} MeV:")
        print(f"  Total: {xcom[i, 1]:.6e}")
        print(f"  Coherent (Rayleigh): {xcom[i, 2]:.6e}")
        print(f"  Incoherent (Compton): {xcom[i, 3]:.6e}")
        print(f"  Photoelectric: {xcom[i, 4]:.6e}")
        print(f"  Pair nuclear: {xcom[i, 5]:.6e}")
        print(f"  Pair electron: {xcom[i, 6]:.6e}")
        print(f"  Total: {xcom[i, 7]:.6e}")
