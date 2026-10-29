import numpy as np

xcom = np.loadtxt('/tmp/data/xcom_water.txt')

# Find row closest to 1.0 MeV
for i in range(len(xcom)):
    if abs(xcom[i, 0] - 1.0) < 0.05:
        print(f"Row {i}: E={xcom[i, 0]:.6f}")
        for j in range(8):
            print(f"  Col {j}: {xcom[i, j]:.6e}")
        
# Also check at 1.25 MeV
print("\nAt ~1.25 MeV:")
for i in range(len(xcom)):
    if abs(xcom[i, 0] - 1.25) < 0.01:
        print(f"Row {i}: E={xcom[i, 0]:.6f}")
        for j in range(8):
            print(f"  Col {j}: {xcom[i, j]:.6e}")
