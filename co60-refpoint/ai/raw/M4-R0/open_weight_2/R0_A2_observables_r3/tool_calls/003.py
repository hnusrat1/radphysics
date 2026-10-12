# Check lines around 1.2 MeV
for line in open('/tmp/data/xcom_water.txt'):
    if '1.17' in line or '1.33' in line:
        print(line.strip())
        
# Also check a wider range
lines = open('/tmp/data/xcom_water.txt').readlines()
for i, line in enumerate(lines):
    if '1.0' in line.split()[0] or '1.5' in line.split()[0]:
        print(f"Line {i}: {line.strip()}")