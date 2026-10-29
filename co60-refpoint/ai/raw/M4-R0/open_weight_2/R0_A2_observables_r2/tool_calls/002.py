import numpy as np

# Read full XCOM data
xcom = np.loadtxt('/tmp/data/xcom_water.txt')
print("XCOM first column (energy in MeV):")
print(xcom[:, 0])
print("\nXCOM columns:")
print("Col 0: Energy (MeV)")
print("Col 1: Total")
print("Col 2: Coherent (Rayleigh)")
print("Col 3: Incoherent (Compton)")
print("Col 4: Photoelectric")
print("Col 5: Pair production (nuclear)")
print("Col 6: Pair production (electron)")
print("Col 7: Total (uncorrected?)")

print("\n---\n")

# Read full ESTAR data  
estar = np.loadtxt('/tmp/data/estar_water.txt')
print("ESTAR columns:")
for i, col in enumerate(estar[0]):
    print(f"Col {i} min/max:", ester[:, i].min(), ester[:, i].max())
    
print("\nESTAR energies:")
print(estar[:, 0])
