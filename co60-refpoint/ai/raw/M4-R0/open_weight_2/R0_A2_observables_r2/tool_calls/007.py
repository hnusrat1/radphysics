import numpy as np

xcom = np.loadtxt('/tmp/data/xcom_water.txt')
print("Full data inspection:")
for i in [0, 10, 20, 30]:
    e = xcom[i, 0]
    print(f"\nRow {i}: E={e:.4f} MeV")
    for j in range(8):
        print(f"  Col {j}: {xcom[i, j]:.6e}")
