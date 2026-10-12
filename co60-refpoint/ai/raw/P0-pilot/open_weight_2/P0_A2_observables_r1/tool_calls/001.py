# Read the data files to understand their structure
import numpy as np

# Read xcom_water.txt
with open('/tmp/data/xcom_water.txt', 'r') as f:
    lines = f.readlines()
    
print("XCOM data header:")
print(lines[0])
print("First few lines:")
for i, line in enumerate(lines[:5]):
    print(f"Line {i}: {line.strip()}")

print("\nNumber of lines:", len(lines))