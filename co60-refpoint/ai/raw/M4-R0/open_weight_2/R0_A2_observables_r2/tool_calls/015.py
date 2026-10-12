import numpy as np

estar = np.loadtxt('/tmp/data/estar_water.txt', comments='#')

# Check radiative vs collision for electrons around 0.3-1 MeV (typical Compton electron energies)
print("Stopping powers for representative electron energies:")
for E in [0.1, 0.3, 0.5, 0.7, 1.0]:
    # Find closest energy
    idx = np.argmin(np.abs(estar[:, 0] - E))
    collision = estar[idx, 1]  # Col 1: collision SP
    radiative = estar[idx, 2]  # Col 2: radiative SP
    total = estar[idx, 3]      # Col 3: total SP
    ratio_rad_col = radiative / collision if collision > 0 else 0
    
    print(f"  E = {E:.2f} MeV:")
    print(f"    Collision SP: {collision:.4f} MeV cm^2/g")
    print(f"    Radiative SP: {radiative:.4e} MeV cm^2/g")
    print(f"    Ratio radiative/collision: {ratio_rad_col:.4e}")
    