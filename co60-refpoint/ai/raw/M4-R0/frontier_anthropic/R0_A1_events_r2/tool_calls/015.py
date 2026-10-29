
import subprocess
t=r'''
import numpy as np
from phys import *
import sim as S
from chordlib import chord
rng=S.rng
Vc=S.Vc
n=400000
Esrc=np.where(rng.random(n)<0.5,1.1732,1.3325)
a=rng.uniform(-0.05,0.05,n); b=rng.uniform(-0.05,0.05,n)
d=np.stack([a,b,np.ones(n)],1); d/=np.linalg.norm(d,axis=1)[:,None]
src=np.array([0,0,-100.])
entry=src+d*(100/d[:,2])[:,None]
pos=entry.copy(); dr=d.copy(); E=Esrc.copy()
gen=0
TLg=[]
first_scatter_depth=[]
while len(E)>0 and gen<12:
    mu=mu_tot(E)
    s=-np.log(rng.random(len(E)))/mu
    dbox=S.box_exit(pos,dr)
    TLg.append((gen, np.sum(mu*chord(pos,dr,np.minimum(s,dbox))), len(E)))
    inside=s<dbox
    pos=pos[inside]; dr=dr[inside]; E=E[inside]; s=s[inside]; mu=mu[inside]
    if len(E)==0: break
    y=pos+s[:,None]*dr
    mc=mu_c(E); mR=mu_R(E); mP=mu_pair(E)
    uu=rng.random(len(E))
    isC=uu<mc/mu
    newp=[];newd=[];newE=[]
    Ei=E[isC]
    epsv=S.KN_eps(Ei)
    cc=1-(1/epsv-1)/(Ei/me)
    dC=S.rotate(dr[isC],cc,2*np.pi*rng.random(len(Ei)))
    pos=y[isC]; dr=dC; E=epsv*Ei
    gen+=1
for g,tl,k in TLg:
    print(g, 'TL (mu*chord) per src photon:', tl/n, 'photons', k/n)
'''
open('t7.py','w').write(t)
r=subprocess.run(['python','t7.py'],capture_output=True,text=True,cwd='.'); print(r.stdout,r.stderr[-2000:])
