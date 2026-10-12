# Check more lines
lines = open('/tmp/data/xcom_water.txt').readlines()
for line in lines:
    parts = line.split()
    if len(parts) >= 8:
        try:
            e = float(parts[0])
            if 0.5 <= e <= 2.0:
                print(f"E={e:.4f}, T={parts[-1]}")
        except: pass