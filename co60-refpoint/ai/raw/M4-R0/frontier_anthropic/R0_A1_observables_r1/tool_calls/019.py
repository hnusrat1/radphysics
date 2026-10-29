
anal = r'''
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
import phys, mc, elec
from phys import re, mc2, ne, Lines, mu_c, mu_pe, mu_pair, mu_coh, mu_tot, ETR, OMEGA
from elec import f_moller, beta2, S_col_BS, S_rad_w, S_res, ne_air
from mc import kn_sample_cos, rotate
PE=[0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4]
EE=[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4]
CE=[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]
def sample_compton(Ev,uv,rng):
    c=kn_sample_cos(Ev,rng); phi=rng.uniform(0,2*np.pi,Ev.size)
    n=rotate(uv,c,phi)
    Eo=Ev/(1+Ev/mc2*(1-c))
    pe=Ev[:,None]*uv-Eo[:,None]*n
    T=Ev-Eo
    return T, pe/np.linalg.norm(pe,axis=1)[:,None]

def build(N, seed, Nel=200000, depths=(0.5,5.0)):
    rng=np.random.default_rng(seed)
    res=mc.run(N,list(depths),rng,store_for=5.0)
    A=res['arr']
    Fn=OMEGA*105.0**2/N
    E=A['E']; u=A['u']; W=A['W']*Fn; typ=A['typ']
    Pw=0.5*np.exp(-mu_tot(Lines)*5.0)   # primary fluence per line per Phi0
    # electron records: lists of arrays
    rec=dict(T=[],D=[],w=[],parent=[],proc=[],charge=[])
    def add(T,D,w,parent,proc,charge):
        rec['T'].append(np.asarray(T,float)); rec['D'].append(np.asarray(D,float)); rec['w'].append(np.asarray(w,float))
        rec['parent'].append(np.full(len(T),parent,int)); rec['proc'].append(np.full(len(T),proc,int)); rec['charge'].append(np.full(len(T),charge,int))
    # proc: 0 compton, 1 photo, 2 pair
    # --- scattered arrivals
    T,D=sample_compton(E,u,rng); add(T,D,W*mu_c(E),0,0,-1)
    add(E.copy(),u.copy(),W*mu_pe(E),0,1,-1)
    Ek=np.clip(E-1.022,0,None); fr=rng.uniform(size=E.size)
    add(Ek*fr,u.copy(),W*mu_pair(E),0,2,-1); add(Ek*(1-fr),u.copy(),W*mu_pair(E),0,2,+1)
    # --- primary lines
    for El,w0 in zip(Lines,Pw):
        Ev=np.full(Nel,El); uv=np.tile([0.0,0.0,1.0],(Nel,1))
        T,D=sample_compton(Ev,uv,rng); add(T,D,np.full(Nel,w0*mu_c(El)/Nel),1,0,-1)
        add([El],[[0,0,1.0]],[w0*mu_pe(El)],1,1,-1)
        if El>1.022:
            Ek1=El-1.022
            add([Ek1*0.5],[[0,0,1.0]],[w0*mu_pair(El)],1,2,-1)
            add([Ek1*0.5],[[0,0,1.0]],[w0*mu_pair(El)],1,2,+1)
    R={k:np.concatenate(v) for k,v in rec.items()}
    R['D']=R['D'] if R['D'].ndim==2 else R['D']
    return dict(R=R, E=E, W=W, u=u, typ=typ, Pw=Pw, Fn=Fn, N=N, res=res)
'''
open('/tmp/work/anal.py','w').write(anal)
import sys; sys.path.insert(0,'/tmp/work')
import time
t=time.time()
import anal, importlib
B=anal.build(100000, 5, Nel=20000)
print(time.time()-t, B['R']['T'].shape, B['R']['D'].shape, B['W'].sum())
