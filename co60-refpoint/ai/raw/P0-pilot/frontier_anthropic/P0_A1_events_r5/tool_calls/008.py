
script = r'''
import sys; sys.path.insert(0,'/tmp/work')
import numpy as np, json
import mc
from collections import Counter
prim,nev,ncoll=mc.simulate(2000000)
def summarize(ev):
    W=ev['W']; return {p: round(float(W[ev['process']==p].sum()/W.sum()),5) for p in ['compton','photoelectric','pair','rayleigh']}
print("primary W",prim['W'].sum(),"scatter W",nev['W'].sum(),"primary frac",prim['W'].sum()/(prim['W'].sum()+nev['W'].sum()))
print("prim",summarize(prim)); print("nee",summarize(nev))
allev={k:np.concatenate([prim[k],nev[k]]) for k in prim}
print("all",summarize(allev))
perm=np.random.default_rng(7).permutation(len(allev['W']))
W=allev['W'][perm]; cw=np.cumsum(W); tot=cw[-1]
step=tot/200; start=np.random.default_rng(11).uniform(0,step)
pts=start+step*np.arange(200)
idx=np.searchsorted(cw,pts)
sel=perm[idx]
print("unique",len(np.unique(sel)))
events=[]
for i in sel:
    p=str(allev['process'][i])
    e={"process":p,"e_in_MeV":round(float(allev['e_in'][i]),5),
       "e_out_MeV":None if np.isnan(allev['e_out'][i]) else round(float(allev['e_out'][i]),5),
       "cos_scatter":None if np.isnan(allev['cos_s'][i]) else round(float(allev['cos_s'][i]),5),
       "t_e_MeV":round(float(allev['t_e'][i]),5),
       "cos_electron":None if np.isnan(allev['cos_e'][i]) else round(float(np.clip(allev['cos_e'][i],-1,1)),5),
       "t_positron_MeV":None if np.isnan(allev['t_pos'][i]) else round(float(allev['t_pos'][i]),5)}
    if p=='rayleigh': e['cos_electron']=None; e['t_e_MeV']=0.0
    if p in ('photoelectric','pair'): e['e_out_MeV']=None; e['cos_scatter']=None
    if p=='photoelectric': e['t_positron_MeV']=None
    if p!='pair': e['t_positron_MeV']=None
    events.append(e)
print(Counter(e['process'] for e in events))
print("e_in=2.0:",sum(1 for e in events if e['e_in_MeV']==2.0))
print("mean e_in", np.mean([e['e_in_MeV'] for e in events]))
out={"method":"placeholder","events":events}
json.dump(out,open('/tmp/work/events.json','w'))
print(json.dumps(events[:5]))
'''
open('/tmp/work/run_all.py','w').write(script)
import subprocess
r=subprocess.run(['python3','/tmp/work/run_all.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-4000:], r.stderr[-3000:])
