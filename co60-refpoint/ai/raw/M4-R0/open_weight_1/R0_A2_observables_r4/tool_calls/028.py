import numpy as np
# patch emc.py: fix CDF interpolation
s=open('/tmp/emc.py').read()
old="""    cdf=np.concatenate([[0],np.cumsum(S*dzg)]); cdf/=cdf[-1]
    zsel=np.interp(rng.uniform(0,1,N_el),cdf,zg)"""
new="""    cdf=np.cumsum(S*dzg); cdf/=cdf[-1]
    zsel=np.interp(rng.uniform(0,1,N_el),cdf,zg)"""
assert old in s
s=s.replace(old,new)
open('/tmp/emc.py','w').write(s)
print("patched")