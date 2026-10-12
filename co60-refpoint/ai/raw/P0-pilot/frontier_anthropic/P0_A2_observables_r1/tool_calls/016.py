
code7 = r'''
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
from phys import *
from deposit import Yrad
ME=0.51099895
NE=3.343e23   # electrons per cm^3 water
K_M=2*np.pi*re*re*ME   # MeV cm^2
X0=36.08
EBe=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
CB=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
# ---- Moller tables: eps grid, E rows
DEL=0.01
EPS=np.exp(np.linspace(np.log(1e-3),np.log(0.5),1500))
ELG=np.exp(np.linspace(np.log(0.0105),np.log(2.5),300))
def moller_f(eps,E):
    tau=E/ME
    return 1/eps**2+1/(1-eps)**2+(tau/(tau+2))**2-(2*tau+1)/((tau+1)**2)/(eps*(1-eps))
def build_moller():
    K=len(ELG); J=len(EPS)
    cdf=np.zeros((K,J)); lam=np.zeros(K)
    for k,E in enumerate(ELG):
        beta2=1-(ME/(E+ME))**2
        epmin=DEL/E
        m=(EPS>=epmin)&(EPS<=0.5)
        if m.sum()<3: continue
        pref=K_M/(beta2*E)
        f=np.where(m, moller_f(EPS,E)*pref, 0.0)
        # cumulative integral over eps (trapezoid), normalized later
        cum=np.concatenate([[0],np.cumsum(0.5*(f[1:]+f[:-1])*np.diff(EPS))])
        lam[k]=cum[-1]*NE
        if cum[-1]>0: cdf[k]=cum/cum[-1]
        else: cdf[k]=0
    return cdf,lam
CDF,LAM=build_moller()
J=len(EPS)
OFF=(CDF+1.5*np.arange(len(ELG))[:,None]).ravel()
def sample_delta_T(E,rng):
    """E: parent energies (array). returns T of delta rays (>= DEL) for given parent; uses nearest ELG row"""
    k=np.clip(np.round((np.log(E)-np.log(ELG[0]))/(np.log(ELG[1])-np.log(ELG[0]))).astype(int),0,len(ELG)-1)
    u=rng.random(len(E))
    g=np.searchsorted(OFF,u+1.5*k)
    j=g-k*J
    j=np.clip(j,1,J-1)
    c0=CDF[k,j-1]; c1=CDF[k,j]
    t=np.where(c1>c0,(u-c0)/np.maximum(c1-c0,1e-300),0.5)
    lg0=np.log(EPS[j-1]); lg1=np.log(EPS[j])
    eps=np.exp(lg0+t*(lg1-lg0))
    return eps*E
def lam_of_E(E):
    k=np.clip(np.round((np.log(E)-np.log(ELG[0]))/(np.log(ELG[1])-np.log(ELG[0]))).astype(int),0,len(ELG)-1)
    return LAM[k]
def rot_dir(u,cth,phi):
    return rotate(u,cth,phi)
def transport(T0,u0,w0,ps0,rng,maxit=5000):
    """electron transport (CSDA + Moller deltas + Highland MS) in infinite medium.
       tallies per unit weight; returns dictionary"""
    E=T0.copy(); u=u0.copy(); w=w0.copy(); ps=ps0.copy(); isd=np.zeros(len(E),bool)
    tb=np.zeros(len(EBe)); tbd=np.zeros(len(EBe)); tcos=np.zeros(10); tcosd=np.zeros(10)
    tot_track=0.0; del_track=0.0; sumE=0.0; sumSc=0.0; sumSr=0.0
    tot_ps_track=0.0
    it=0
    nd_created=0
    # losses for dose check
    Lcol=0.0
    while len(E)>0 and it<maxit:
        it+=1
        keep=E>=DEL
        if not keep.all():
            E=E[keep];u=u[keep];w=w[keep];ps=ps[keep];isd=isd[keep]
            if len(E)==0: break
        S=Stot(E); Sc=Scol(E); Sr=Srad(E)
        dE=0.03*E
        ds=dE/S
        Em=E-dE/2
        # tallies
        wd=w*ds
        bi=np.digitize(Em,EBe)-1   # -1 below 0.01 (never), 25 = overflow
        ok=(bi>=0)&(bi<25)
        np.add.at(tb,bi[ok],wd[ok])
        np.add.at(tbd,bi[ok],wd[ok]*isd[ok])
        cb=np.clip(np.digitize(u[:,2],CB)-1,0,9)
        np.add.at(tcos,cb,wd)
        np.add.at(tcosd,cb,wd*isd)
        tot_track+=wd.sum(); del_track+=(wd*isd).sum(); sumE+=(wd*Em).sum()
        sumSc+=(wd*Sc).sum(); sumSr+=(wd*Sr).sum()
        # deltas
        lam=lam_of_E(E)*ds
        nd=rng.poisson(lam)
        idx=np.repeat(np.arange(len(E)),nd)
        # parents' energies after step
        Enew=E-dE
        # multiple scattering on parents
        nd_tot=len(idx)
        if nd_tot>0:
            Tp=sample_delta_T(E[idx],rng)
            # keep T < E/2 (Moller max) automatically; also T< dE allowed
            Tsum=np.zeros(len(E)); np.add.at(Tsum,idx,Tp)
            cth=np.sqrt(np.clip(Tp*(E[idx]+2*ME)/(E[idx]*(Tp+2*ME)),0,1))
            phi=rng.uniform(0,2*np.pi,nd_tot)
            ud=rotate(u[idx],cth,phi)
            # parent direction update by momentum conservation
            pp=np.sqrt(E*(E+2*ME)); pd=np.sqrt(Tp*(Tp+2*ME))
            # update parent direction for each parent with deltas: approximate using summed momentum
            pvec=np.zeros_like(u); np.add.at(pvec,idx,pd[:,None]*ud)
            newp=pp[:,None]*u-pvec
            nrm=np.linalg.norm(newp,axis=1)
            good=(nrm>0)&(np.repeat(True,len(E)))
            u=np.where((nrm>1e-12)[:,None], newp/np.maximum(nrm,1e-300)[:,None], u)
            # new delta electrons
            Dw=w[idx]; Dps=ps[idx]
            nd_created+=nd_tot
        else:
            Tp=np.zeros(0)
        # Multiple scattering for parents (after momentum update) 
        x=ds
        pc=np.sqrt(E*(E+2*ME)); beta=pc/(E+ME)
        th0=13.6/(beta*pc)*np.sqrt(np.maximum(x,1e-30)/X0)*(1+0.038*np.log(np.maximum(x,1e-30)/X0))
        th0=np.clip(th0,0,np.pi)
        e1,e2=_orth(u)
        g1=rng.normal(size=len(E))*th0; g2=rng.normal(size=len(E))*th0
        un=u+g1[:,None]*e1+g2[:,None]*e2
        u=un/np.linalg.norm(un,axis=1,keepdims=True)
        # Enew
        E=Enew
        isd_p=isd
        # append deltas
        if nd_tot>0:
            # delta electrons: energies Tp, directions ud (already computed before ms, fine)
            E=np.concatenate([E,Tp]); u=np.concatenate([u,ud]); w=np.concatenate([w,Dw]); ps=np.concatenate([ps,Dps]); isd=np.concatenate([isd,np.ones(nd_tot,bool)])
    Lcol=sumSc
    return dict(tb=tb,tbd=tbd,tcos=tcos,tcosd=tcosd,tot=tot_track,delt=del_track,sumE=sumE,Sc=sumSc,Sr=sumSr,nd=nd_created,it=it)
def _orth(d):
    a=np.where(np.abs(d[:,0:1])<0.9, np.array([[1.,0,0]]), np.array([[0,1.,0]]))
    e1=np.cross(d,a); e1/=np.linalg.norm(e1,axis=1,keepdims=True)
    e2=np.cross(d,e1)
    return e1,e2
'''
open('/tmp/work/electrons.py','w').write(code7)
import sys, importlib, time
sys.path.insert(0,'/tmp/work')
import electrons
print("LAM sample", electrons.LAM[::60])
rng=np.random.default_rng(3)
t=time.time()
r=electrons.transport(np.full(2000,1.0),np.tile([[0,0,1.]],(2000,1)),np.ones(2000),np.zeros(2000,bool),rng)
print(time.time()-t, r['it'], r['tot']/2000, r['delt']/r['tot'], r['nd']/2000)
