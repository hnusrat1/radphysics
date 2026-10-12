import numpy as np

# Load and examine XCOM data more carefully
xcom = np.loadtxt('/tmp/data/xcom_water.txt')
print("XCOM column headers from file header:")
with open('/tmp/data/xcom_water.txt', 'r') as f:
    for line in f:
        if line.startswith('#'):
            print(line.strip())

print("\nLast few rows of xcom:")
for i in range(-5, 0):
    print(xcom[i])
