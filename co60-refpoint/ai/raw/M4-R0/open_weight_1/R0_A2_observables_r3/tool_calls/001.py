import numpy as np, os
for f in ['/tmp/data/xcom_water.txt','/tmp/data/estar_water.txt']:
    print('='*80); print(f); print(open(f).read()[:1500])