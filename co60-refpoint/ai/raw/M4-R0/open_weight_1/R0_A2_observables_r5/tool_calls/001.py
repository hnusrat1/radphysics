import numpy as np, os
for f in ['/tmp/data/xcom_water.txt','/tmp/data/estar_water.txt']:
    print('='*80)
    print(f)
    with open(f) as fh:
        for i,l in enumerate(fh):
            if i<15: print(l.rstrip())
            else: break
    print('...')