import numpy as np
import os

# Check the data files
print("Files available:")
for f in os.listdir('/tmp/data'):
    print(f)

print("\n\n=== XCOM WATER ===")
with open('/tmp/data/xcom_water.txt', 'r') as f:
    content = f.read()
print(content[:2000])

print("\n\n=== ESTAR WATER ===")
with open('/tmp/data/estar_water.txt', 'r') as f:
    content = f.read()
print(content[:2000])