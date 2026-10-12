
import importlib, mc, numpy as np, json, time
importlib.reload(mc)
t0=time.time()
prim,nev,ncoll=mc.simulate(2000000)
print("time",time.time()-t0)
def summarize(ev):
    W=ev['W']; out={}
    for p in ['compton','photoelectric','pair','rayleigh']:
        out[p]=W[ev['process']==p].sum()/W.sum()
    return out
print("primary W",prim['W'].sum(),"scatter W",nev['W'].sum(), "primary frac",prim['W'].sum()/(prim['W'].sum()+nev['W'].sum()))
print("prim process frac",summarize(prim))
print("nee process frac",summarize(nev))
# combined
allev={k:np.concatenate([prim[k],nev[k]]) for k in prim}
print("all process frac",summarize(allev))
