
import numpy as np, sys, glob
sys.path.insert(0,'/tmp/work'); import lib
Om=lib.Om_p; fac=Om*105**2
for f in sorted(glob.glob('/tmp/work/H_*.npz')):
    d=np.load(f); H=d['H']; N=float(d['N']); dep=d['depths']
    print(f, N, [round(H[j].sum()/N*fac,4) for j in range(len(dep))])
