
mc = r'''
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
import xs
ME=0.51099895
RE=2.8179403e-13
SRC_Z=-100.0; DIST=100.0
OMEGA=0.0099751  # sr of 10x10 at 100 cm (computed exactly)
R0=110.0
PHI0_PER_SRC=1.0/(OMEGA*R0**2)
CYL_R=1.0; CYL_Z0=9.9; CYL_Z1=10.1
VCYL=np.pi*CYL_R**2*0.2
# energy grids for tallies
EB=np.arange(0,2.0001,0.01)   # fine bins 0.01 MeV
# KN mu-tables for Compton sampling
def kn_fk(E):
    # energy transfer fraction f_KN = <1-E'/E> / ... (mean fraction of E transferred) computed via quadrature
    e=E/ME
    mu=np.linspace(-1,1,4001)
    k=1/(1+e*(1-mu))
    w=k**2*(k+1/k-(1-mu**2))
    sig=np.trapz(w,mu)
    tr=np.trapz(w*(1-k),mu)
    return tr/sig
# precompute CDF in mu for KN at log-spaced E grid
EG=np.exp(np.linspace(np.log(0.01),np.log(2.0),400))
MUG=np.linspace(-1,1,2001)
CDFG=np.zeros((len(EG),len(MUG)))
for i,E in enumerate(EG):
    e=E/ME; k=1/(1+e*(1-MUG)); w=k**2*(k+1/k-(1-MUG**2))
    c=np.concatenate([[0],np.cumsum(0.5*(w[1:]+w[:-1])*np.diff(MUG))])
    CDFG[i]=c/c[-1]
def kn_sample_mu(E,u):
    # E array; u uniform array
    lE=np.log(np.clip(E,EG[0],EG[-1]))
    x=(lE-np.log(EG[0]))/(np.log(EG[-1])-np.log(EG[0]))*(len(EG)-1)
    i0=np.clip(np.floor(x).astype(int),0,len(EG)-2); f=x-i0
    # interpolate CDF rows then invert
    cdf=(1-f)[:,None]*CDFG[i0]+f[:,None]*CDFG[i0+1]
    # invert per row via searchsorted (vectorized by cumulative count)
    idx=np.array([np.searchsorted(cdf[j],u[j]) for j in range(len(u))]) if len(u)<0 else None
    # vectorized inversion
    j=(cdf<u[:,None]).sum(axis=1)
    j=np.clip(j,1,len(MUG)-1)
    c0=cdf[np.arange(len(u)),j-1]; c1=cdf[np.arange(len(u)),j]
    t=(u-c0)/np.maximum(c1-c0,1e-300)
    return MUG[j-1]+t*(MUG[j]-MUG[j-1])
def ortho(d):
    ax=np.where(np.abs(d[:,0:1])<0.9, np.array([[1.0,0,0]]), np.array([[0,1.0,0]]))
    e1=np.cross(d,ax); e1/=np.linalg.norm(e1,axis=1)[:,None]
    e2=np.cross(d,e1)
    return e1,e2
def rotate(d,mu,phi):
    e1,e2=ortho(d)
    s=np.sqrt(np.maximum(0,1-mu**2))
    nd=mu[:,None]*d+s[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)
    return nd/np.linalg.norm(nd,axis=1)[:,None]
def rayleigh_mu(n,rng):
    out=np.empty(n)
    todo=np.arange(n)
    while len(todo):
        m=rng.uniform(-1,1,len(todo))
        acc=rng.uniform(0,1,len(todo))< (1+m**2)/2.0
        out[todo[acc]]=m[acc]; todo=todo[~acc]
    return out
def chord_overlap(p,d,t_hi):
    # length in [0,t_hi] of segment p+t d inside cylinder radius 1, z in [9.9,10.1]
    ta=np.zeros(len(p)); tb=t_hi.copy()
    dz=d[:,2]; z0=p[:,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        t1=(CYL_Z0-z0)/dz; t2=(CYL_Z1-z0)/dz
    lo=np.where(dz>0,t1,t2); hi=np.where(dz>0,t2,t1)
    zpar=np.abs(dz)<1e-12
    inside_z=(z0>=CYL_Z0)&(z0<=CYL_Z1)
    lo=np.where(zpar,np.where(inside_z,-np.inf,np.inf),lo)
    hi=np.where(zpar,np.where(inside_z,np.inf,-np.inf),hi)
    a=d[:,0]**2+d[:,1]**2; b=2*(p[:,0]*d[:,0]+p[:,1]*d[:,1]); c=p[:,0]**2+p[:,1]**2-CYL_R**2
    disc=b*b-4*a*c
    with np.errstate(divide='ignore',invalid='ignore'):
        sq=np.sqrt(np.maximum(disc,0))
        r1=(-b-sq)/(2*a); r2=(-b+sq)/(2*a)
    rlo=np.where(a>1e-14, np.where(disc>=0, r1, np.inf), -np.inf)
    rhi=np.where(a>1e-14, np.where(disc>=0, r2, -np.inf), np.inf)
    # if a tiny and inside disc: ok
    L_lo=np.maximum.reduce([np.zeros(len(p)),lo,rlo])
    L_hi=np.minimum.reduce([t_hi,hi,rhi])
    return np.maximum(L_hi-L_lo,0.0), L_lo, L_hi
def ftr(E): 
    return np.array([kn_fk(e) for e in np.atleast_1d(E)]) if False else None
'''
open('/tmp/work/mcph.py','w').write(mc)
import importlib,sys; sys.path.insert(0,'/tmp/work')
import mcph
print(mcph.kn_sample_mu(np.array([2.0,0.1]),np.array([0.5,0.5])))
# test chord overlap
p=np.array([[0,0,0.0],[0.5,0,0.0]]); d=np.array([[0,0,1.0],[0,0,1.0]])
print(mcph.chord_overlap(p,d,np.array([30.,30.])))
