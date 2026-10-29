# Check if there are headers in the files
with open('/tmp/data/xcom_water.txt', 'r') as f:
    lines = f.readlines()
    print("XCOM file first 10 lines:")
    for i, line in enumerate(lines[:10]):
        print(f"Line {i}: {line.strip()}")
        
print("\n---\n")

with open('/tmp/data/estar_water.txt', 'r') as f:
    lines = f.readlines()
    print("ESTAR file first 10 lines:")
    for i, line in enumerate(lines[:10]):
        print(f"Line {i}: {line.strip()}")
