import numpy as np

estar = np.loadtxt('/tmp/data/estar_water.txt')
print("ESTAR data shape:", ester.shape)
print("\nColumn descriptions based on file name:")
print("Col 0: Electron kinetic energy (MeV)")
print("Col 1: Collision stopping power (MeV cm^2/g)")
print("Col 2: Radiative stopping power (MeV cm^2/g)") 
print("Col 3: Total stopping power (MeV cm^2/g)")
print("Col 4: CSDA range (g/cm^2)")
print("Col 5: Radiation yield")
print("Col 6: Density-effect parameter")

print("\nEnergy range:", ester[:, 0].min(), "to", ester[:, 0].max(), "MeV")
print("\nStopping powers at ~1.2 MeV:")
for i, row in enumerate(estar):
    if 1.15 <= row[0] <= 1.2:
        print(f"E={row[0]:.4f} MeV:")
        print(f"  Collision: {row[1]:.4e}, Radiative: {row[2]:.4e}, Total: {row[3]:.4e}")
        break
