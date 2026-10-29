
allev={k:np.concatenate([prim[k],nev[k]]) for k in prim}
print("all process frac",summarize(allev))
perm=rng=np.random.default_rng(7).permutation(len(allev['W']))
W=allev['W'][perm]; cw=np.cumsum(W); tot=cw[-1]
step=tot/200; start=np.random.default_rng(11).uniform(0,step)
pts=start+step*np.arange(200)
idx=np.searchsorted(cw,pts)
sel=perm[idx]
print("unique selected",len(np.unique(sel)))
events=[]
def f(x):
    return None if (x is None or (isinstance(x,float) and np.isnan(x))) else float(x)
for i in sel:
    p=str(allev['process'][i])
    e={"process":p,"e_in_MeV":round(float(allev['e_in'][i]),5),
       "e_out_MeV":None if np.isnan(allev['e_out'][i]) else round(float(allev['e_out'][i]),5),
       "cos_scatter":None if np.isnan(allev['cos_s'][i]) else round(float(allev['cos_s'][i]),5),
       "t_e_MeV":round(float(allev['t_e'][i]),5),
       "cos_electron":None if np.isnan(allev['cos_e'][i]) else round(float(np.clip(allev['cos_e'][i],-1,1)),5),
       "t_positron_MeV":None if np.isnan(allev['t_pos'][i]) else round(float(allev['t_pos'][i]),5)}
    if p=='rayleigh': e['cos_electron']=None; e['t_e_MeV']=0.0
    if p=='photoelectric': e['e_out_MeV']=None; e['cos_scatter']=None
    if p=='pair': e['e_out_MeV']=None; e['cos_scatter']=None
    if p=='compton' and e['cos_electron'] is None: pass
    events.append(e)
from collections import Counter
print(Counter(e['process'] for e in events))
print("primary-type (e_in=2.0):",sum(1 for e in events if e['e_in_MeV']==2.0))
print(events[:3])
print(events[-3:])
