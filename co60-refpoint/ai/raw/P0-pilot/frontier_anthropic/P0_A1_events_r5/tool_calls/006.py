
import sys; sys.path.insert(0,'/tmp/work')
import importlib, numpy as np, json, time
import mc
t0=time.time()
prim,nev,ncoll=mc.simulate(2000000)
print("time",time.time()-t0)
def summarize(ev):
    W=ev['W']; out={}
    for p in ['compton','photoelectric','pair','rayleigh']:
        out[p]=round(float(W[ev['process']==p].sum()/W.sum()),5)
    return out
print("primary W",prim['W'].sum(),"scatter W",nev['W'].sum(), "primary frac",prim['W'].sum()/(prim['W'].sum()+nev['W'].sum()))
print("prim process frac",summarize(prim))
print("nee process frac",summarize(nev))
