import numpy as np

with open('/tmp/data/estar_water.txt', 'r') as f:
    estar_lines = f.readlines()

print(f"Total lines: {len(estar_lines)}")
print("\nFirst 10 lines:")
for i, line in enumerate(estar_lines[:10]):
    print(f"{i}: {repr(line[:80])}")

# Try parsing differently - maybe comma separated?
data = []
for i, line in enumerate(estar_lines):
    if i < 4:  # Skip header
        continue
    parts = line.strip().split()
    if len(parts) >= 8:
        try:
            vals = [float(p) for p in parts]
            data.append(vals)
        except:
            pass
            
print(f"\nParsed {len(data)} rows")
if len(data) > 0:
    print("First row:", data[0])
    print("Last row:", data[-1])