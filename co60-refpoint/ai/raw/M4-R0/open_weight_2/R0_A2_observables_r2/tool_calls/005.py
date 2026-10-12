import numpy as np
data_e = np.loadtxt('/tmp/data/estar_water.txt')
print("Shape:", data_e.shape)
print("First row:", data_e[0])
print("Last row:", data_e[-1])
print("\nColumns at E~1.25 MeV:")
for i in range(len(data_e)):
    if 1.2 <= data_e[i, 0] <= 1.3:
        print(f"Row {i}: E={data_e[i, 0]:.4f}")
        print(f"  Collision SP: {data_e[i, 1]:.6f}")
        print(f"  Radiative SP: {data_e[i, 2]:.6e}")
        print(f"  Total SP: {data_e[i, 3]:.6f}")
        break
