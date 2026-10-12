
code6 = r'''
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
from phys import *
from deposit import transfer_coeffs, Yrad
Om=0.009975072683099257; S_phi0=Om*110.0**2; V=np.pi*1.0**2*0.2
EBe=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
CB=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
def build_sources(seg, Ntot, rng, Phi_p, E_p=2.0):
    """returns list of electron samples: arrays T, uvec, w, parent_scat(bool), charge (-1 e-, +1 e+), kind"""
    out=[]
    # --- primary sources (from primary photons in cylinder: rate Phi_p * mu_i(2.0))
    coh,inc,pe,pr,mt=mu_all(np.array([E_p]))
    Nps=200000
    rate_c=Phi_p*inc[0]; rate_pe=Phi_p*pe[0]; rate_pr=Phi_p*pr[0]
    d0=np.tile(np.array([[0.,0.,1.]]),(Nps,1))
    ep,ct=sample_KN(np.full(Nps,E_p),rng)
    phi=rng.uniform(0,2*np.pi,Nps)
    dsc=rotate(d0,ct,phi)
    pe_vec=E_p*d0-E_p*ep[:,None]*dsc
    Te=E_p*(1-ep)
    ue=pe_vec/np.linalg.norm(pe_vec,axis=1,keepdims=True)
    wc=np.full(Nps,rate_c/Nps)
    out.append(dict(T=Te,u=ue,w=wc,ps=np.zeros(Nps,bool),q=-np.ones(Nps),kind=np.zeros(Nps,int),par=np.zeros(Nps,int)))
    # primary pair
    Npp=20000
    if rate_pr>0:
        u=rng.uniform(0,1,Npp); Tm=(E_p-2*0.511)
        Te1=u*Tm; Tp1=(1-u)*Tm
        d0p=np.tile(np.array([[0.,0.,1.]]),(Npp,1))
        out.append(dict(T=Te1,u=d0p,w=np.full(Npp,rate_pr/Npp),ps=np.zeros(Npp,bool),q=-np.ones(Npp),kind=np.ones(Npp,int),par=np.zeros(Npp,int)))
        out.append(dict(T=Tp1,u=d0p,w=np.full(Npp,rate_pr/Npp),ps=np.zeros(Npp,bool),q=np.ones(Npp),kind=np.ones(Npp,int),par=np.zeros(Npp,int)))
    if rate_pe>0:
        out.append(dict(T=np.array([E_p]),u=np.array([[0.,0.,1.]]),w=np.array([rate_pe]),ps=np.array([False]),q=np.array([-1.]),kind=np.array([2]),par=np.array([0])))
    # --- scattered sources (segments)
    E=seg['E']; d=seg['d']; l=seg['l']
    n=len(E)
    coh,inc,pe,pr,mt=mu_all(E)
    wgt = l*S_phi0/(V*Ntot)  # per seg, per cm3 per Phi0, times mu below
    # Compton
    ep,ct=sample_KN(E,rng)
    phi=rng.uniform(0,2*np.pi,n)
    dsc=rotate(d,ct,phi)
    pev=E[:,None]*d-(E*ep)[:,None]*dsc
    Te=E*(1-ep)
    ue=pev/np.linalg.norm(pev,axis=1,keepdims=True)
    out.append(dict(T=Te,u=ue,w=wgt*inc,ps=np.ones(n,bool),q=-np.ones(n),kind=np.zeros(n,int),par=np.ones(n,int)))
    # photoelectric
    out.append(dict(T=E.copy(),u=d.copy(),w=wgt*pe,ps=np.ones(n,bool),q=-np.ones(n),kind=np.full(n,2),par=np.ones(n,int)))
    # pair (scattered parents)
    ok=E>1.022
    Tm=np.where(ok,E-1.022,0.0)
    uu=rng.uniform(0,1,n)
    out.append(dict(T=Tm*uu,u=d.copy(),w=np.where(ok,wgt*pr,0.0),ps=np.ones(n,bool),q=-np.ones(n),kind=np.ones(n,int),par=np.ones(n,int)))
    out.append(dict(T=Tm*(1-uu),u=d.copy(),w=np.where(ok,wgt*pr,0.0),ps=np.ones(n,bool),q=np.ones(n),kind=np.ones(n,int),par=np.ones(n,int)))
    cat={k:np.concatenate([o[k] for o in out]) for k in ['T','u','w','ps','q','kind','par']}
    return cat
'''
open('/tmp/work/sources.py','w').write(code6)
import sys, importlib, pickle
sys.path.insert(0,'/tmp/work')
import numpy as np
import sources
D=pickle.load(open('/tmp/work/photon_tallies.pkl','rb'))
seg=D['seg']; tot=D['tot']
Ntot=4000000
rng=np.random.default_rng(5)
from phys import mu_all
Phi_p=np.exp(-mu_all(np.array([2.0]))[4][0]*10)
cat=sources.build_sources(seg,Ntot,rng,Phi_p)
T=cat['T'];w=cat['w']
print("n samples",len(T),"total rate (per cm3 per Phi0, all incl. e+)",w.sum())
