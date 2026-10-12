import numpy as np

# Load data
def load_data(path):
    cols = np.loadtxt(path, delimiter=None, skiprows=0, unpack=True) # first line might be header
    return cols

# Read xcom_water.txt
xcom_lines = open('/tmp/data/xcom_water.txt').readlines()
xcom_data = []
for line in xcom_lines:
    parts = line.split()
    if len(parts) >= 2:
        try:
            energies = [float(x) for x in parts]
            xcom_data.append(energies)
        except: pass

# Parse xcom
# Format typically: E, coherent, incoherent, photoelectric, pair_nuc, pair_elec, total
# Let's print first few lines to see structure
print("XCOM head:", xcom_data[:2])