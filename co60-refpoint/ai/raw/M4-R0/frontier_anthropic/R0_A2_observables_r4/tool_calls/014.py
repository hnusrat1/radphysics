
import sys; sys.path.insert(0,'/tmp/work')
import mc, mc2, numpy as np, time
from scipy.integrate import dblquad
Om=0.009975072683099257
vals_old=[];vals_new=[]
t=time.time()
for s in range(3):
    r=mc.run(200000,100+s)
    vals_old.append(r['g'][r['det']==4].sum()*Om*105**2/r['N'])
    S=mc2.run(200000,200+s)
    vals_new.append(S['g'][4]*Om*105**2/S['N'])
print(vals_old, vals_new, time.time()-t)
