import numpy as np

xcom = np.loadtxt('/tmp/data/xcom_water.txt')
print("XCOM shape:", xcom.shape)
print("First row:", xcom[0])
print("Last row:", xcom[-1])
print("Columns (guess):", ['E', 'rayleigh', 'coherent', 'incoherent', 'photoelectric', 'pair_nuc', 'pair_el', 'total'])

# Let's check header if any
with open('/tmp/data/xcom_water.txt', 'r') as f:
    lines = f.readlines()
    print("Header:", lines[0].strip())
