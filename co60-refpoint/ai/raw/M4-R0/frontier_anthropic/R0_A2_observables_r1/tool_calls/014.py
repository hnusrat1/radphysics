
import numpy as np, sys
sys.path.insert(0,'/tmp/work'); import lib
d=np.load('/tmp/work/H_test.npz')
H=d['H']; N=float(d['N']); dep=d['depths']
Om=lib.Om_p; fac=Om*105**2
for j,z in enumerate(dep):
    prim=0.5*(np.exp(-lib.mutot(np.array([1.1732]))[0]*z)+np.exp(-lib.mutot(np.array([1.3325]))[0]*z))*(105/(100+z))**2
    s=H[j].sum()/N*fac
    print(f"z={z}: scattered/Phi0={s:.4f}, primary/Phi0={prim:.4f}, S/P={s/prim:.3f}")
