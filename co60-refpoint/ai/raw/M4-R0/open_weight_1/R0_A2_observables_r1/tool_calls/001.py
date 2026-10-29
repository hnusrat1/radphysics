import numpy as np, os
for f in ['/tmp/data/xcom_water.txt','/tmp/data/estar_water.txt']:
    print('='*80)
    print(f, os.path.getsize(f), 'bytes')
    with open(f) as fh:
        for i in range(40):
            line = fh.readline()
            print(line.rstrip())
            if i>38: break