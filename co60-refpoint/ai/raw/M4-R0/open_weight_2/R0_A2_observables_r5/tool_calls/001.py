# First, let's examine the data files
import os

# Read xcom_water.txt
with open('/tmp/data/xcom_water.txt', 'r') as f:
    xcom_lines = f.readlines()
    
print("XCOM file header:")
print(xcom_lines[0])
print("\nFirst 5 data rows:")
for line in xcom_lines[1:6]:
    print(line.strip())
print(f"\nTotal lines: {len(xcom_lines)}")

# Read estar_water.txt
with open('/tmp/data/estar_water.txt', 'r') as f:
    estar_lines = f.readlines()

print("\n\nESTAR file header:")
print(estar_lines[0])
print("\nFirst 5 data rows:")
for line in estar_lines[1:6]:
    print(line.strip())
print(f"\nTotal lines: {len(estar_lines)}")