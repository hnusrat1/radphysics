
import subprocess
t=r'''
import numpy as np, json
from phys import *
import sim as S
rng=np.random.default_rng(777)
C=np.load('C.npy')
perm=rng.permutation(len(C)); C=C[perm]
w=C[:,1]; W=w.sum(); cum=np.cumsum(w)
u0=rng.random()
pts=(u0+np.arange(200))*W/200
idx=np.searchsorted(cum,pts)
sel=C[idx]
assert len(np.unique(idx))==200
events=[]
for typ,_,Earr in sel:
    Earr=float(Earr); typ=int(typ)
    mc=float(mu_c(Earr)); mR=float(mu_R(Earr)); mP=float(mu_pair(Earr)); mpe=float(mu_pe(Earr))
    mt=mc+mR+mP+mpe
    # photon at x: process selection
    r=rng.random()*mt
    if r<mc: proc='compton'
    elif r<mc+mR: proc='rayleigh'
    elif r<mc+mR+mP: proc='pair'
    else: proc='photoelectric'
    if proc=='compton':
        eps=float(S.KN_eps(np.array([Earr]))[0]); a=Earr/me
        cth=1-(1/eps-1)/a
        Eout=eps*Earr
        th2=np.arccos(np.clip(cth,-1,1))/2
        tt=np.tan(th2)
        cosphi=(1+a)*tt/np.sqrt(1+((1+a)*tt)**2) if tt<1e6 else 1.0
        cos_e=float(cosphi)
        ev=dict(process='compton',e_in_MeV=Earr,e_out_MeV=Eout,cos_scatter=float(cth),
                t_e_MeV=Earr-Eout,cos_electron=cos_e,t_positron_MeV=None)
    elif proc=='rayleigh':
        cth=float(S.rayl_sample(np.array([Earr]))[0])
        ev=dict(process='rayleigh',e_in_MeV=Earr,e_out_MeV=Earr,cos_scatter=cth,
                t_e_MeV=0.0,cos_electron=None,t_positron_MeV=None)
    elif proc=='pair':
        avail=Earr-2*me
        te=avail*rng.random(); tp=avail-te
        sig=me/(te+me)
        th=abs(rng.normal(0,sig))
        ev=dict(process='pair',e_in_MeV=Earr,e_out_MeV=None,cos_scatter=None,
                t_e_MeV=float(te),cos_electron=float(np.cos(th)),t_positron_MeV=float(tp))
    else:
        ev=dict(process='photoelectric',e_in_MeV=Earr,e_out_MeV=None,cos_scatter=None,
                t_e_MeV=Earr,cos_electron=float(0.5+0.5*rng.random()),t_positron_MeV=None)
    events.append(ev)
from collections import Counter
print(Counter(e['process'] for e in events))
print(Counter(round(e['e_in_MeV'],4) for e in events).most_common(8))
print('mean e_in', np.mean([e['e_in_MeV'] for e in events]))
print('pair in sample: ', [e for e in events if e['process']=='pair'][:3])
out={"method": "", "events": events}
json.dump(out, open('events.json','w'))
'''
open('sample.py','w').write(t)
r=subprocess.run(['python','sample.py'],capture_output=True,text=True,cwd='.'); print(r.stdout,r.stderr[-3000:])
