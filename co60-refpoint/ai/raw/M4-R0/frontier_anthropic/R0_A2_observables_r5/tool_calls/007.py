
mcmod = r'''
import numpy as np
from phys import *

PI=np.pi
Omega = 4*np.arcsin(0.0025/1.0025)   # sr, 10x10 cm field at 100 cm
Phi0_src = 1.0/(Omega*105.0**2)      # Phi0 per emitted photon (both lines)
LINES = np.array([1.1732,1.3325])
BOX_LO=np.array([-15.,-15.,0.]); BOX_HI=np.array([15.,15.,30.])
ECUT=0.01

# energy-transfer / energy-absorption coefficients (per cm) vs photon energy (KN-based)
_eg = np.logspace(np.log10(0.0102),np.log10(1.42),200)
def _kn_avg(E):
    k=E/mc2
    c=np.linspace(-1,1,4001)
    eps=1/(1+k*(1-c)); sin2=1-c*c
    f = eps**2*(eps+1/eps-sin2)   # proportional to dsigma/dOmega
    T = E*(1-eps)
    Y = Yld(np.maximum(T,0.01))
    w = f*np.ones_like(c)
    norm=np.trapz(w,c)
    ftr = np.trapz(w*(T/E),c)/norm
    fen = np.trapz(w*(T/E)*(1-Y),c)/norm
    return ftr, fen
_ftr=[];_fen=[]
for E in _eg:
    a,b=_kn_avg(E); _ftr.append(a); _fen.append(b)
_ftr=np.array(_ftr); _fen=np.array(_fen)
def mu_tr_c(E): return mu_inc(E)*np.interp(np.log(E),np.log(_eg),_ftr)
def mu_en_c(E): return mu_inc(E)*np.interp(np.log(E),np.log(_eg),_fen)
def mu_tr_pe(E): return mu_pe(E)
def mu_en_pe(E): return mu_pe(E)*(1-Yld(np.maximum(E,0.01)))
def mu_tr_pair(E): return mu_pair(E)*np.clip(1-1.022/np.maximum(E,1.022),0,1)
def mu_en_pair(E): return mu_tr_pair(E)*(1-Yld(np.maximum(E-1.022,0.01)*0.5+0.0))

def sample_kn(E, rng):
    """Klein-Nishina: returns eps=E'/E and cos(theta) (vectorized rejection)."""
    E=np.asarray(E,float); n=E.size
    k=E/mc2
    emin=1/(1+2*k); Lg=np.log(1+2*k); Bn=(1-emin**2)/2; a=Lg/(Lg+Bn)
    eps=np.empty(n); cs=np.empty(n)
    todo=np.arange(n)
    while todo.size:
        kk=k[todo]; em=emin[todo]
        r1=rng.random(todo.size); r2=rng.random(todo.size); r3=rng.random(todo.size)
        A = r1 < a[todo]
        ep = np.where(A, em**r2, np.sqrt(em**2+r2*(1-em**2)))
        c = 1-(1/ep-1)/kk
        s2 = 1-c*c
        acc = r3 < (ep+1/ep-s2)/(ep+1/ep)
        idx=todo[acc]
        eps[idx]=ep[acc]; cs[idx]=c[acc]
        todo=todo[~acc]
    return eps, cs

def rot_dir(d, cs, rng):
    """rotate unit vectors d (n,3) by polar cos cs and random azimuth"""
    n=d.shape[0]
    phi=2*PI*rng.random(n)
    sn=np.sqrt(np.maximum(0,1-cs*cs))
    # build perpendicular basis
    a=np.where(np.abs(d[:,0:1])<0.9, np.array([[1.0,0,0]]), np.array([[0,1.0,0]]))
    e1=np.cross(d,a); e1/=np.linalg.norm(e1,axis=1)[:,None]
    e2=np.cross(d,e1)
    return cs[:,None]*d + sn[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)

def iso_dir(n, rng):
    c=2*rng.random(n)-1; ph=2*PI*rng.random(n); s=np.sqrt(1-c*c)
    return np.column_stack([s*np.cos(ph), s*np.sin(ph), c])

def sample_rayleigh_cos(n, rng):
    out=np.empty(n); todo=np.arange(n)
    while todo.size:
        c=2*rng.random(todo.size)-1; r=rng.random(todo.size)
        acc = r < (1+c*c)/2
        out[todo[acc]]=c[acc]; todo=todo[~acc]
    return out

def cyl_sample(rng, n, zlo, zhi, R=1.0):
    r = R*np.sqrt(rng.random(n)); ph=2*PI*rng.random(n)
    z = zlo + (zhi-zlo)*rng.random(n)
    return np.column_stack([r*np.cos(ph), r*np.sin(ph), z])

def electrons_from_arrivals(Ep, om, val, rng, tag):
    """Generate initial electrons from photon arrivals (E', unit dir om, weight val = fluence contribution).
    Returns list of (T, cos_e, weight, is_positron, is_primary_flag=tag)"""
    out=[]
    n=Ep.size
    if n==0: return out
    # Compton
    mi = mu_inc(Ep)
    eps, cs = sample_kn(Ep, rng)
    d2 = rot_dir(om, cs, rng)
    pe_vec = Ep[:,None]*om - (Ep*eps)[:,None]*d2
    cose = pe_vec[:,2]/np.maximum(np.linalg.norm(pe_vec,axis=1),1e-30)
    T = Ep*(1-eps)
    out.append((T, cose, val*mi, np.zeros(n,bool), tag))
    # photoelectric
    mp = mu_pe(Ep)
    out.append((Ep.copy(), om[:,2].copy(), val*mp, np.zeros(n,bool), tag))
    # pair
    mpr = mu_pair(Ep)
    ok = Ep>1.022
    if ok.any():
        Tp = Ep[ok]-1.022; U=rng.random(ok.sum())
        w = val[ok]*mpr[ok]
        out.append((U*Tp, om[ok,2], w, np.zeros(ok.sum(),bool), tag))   # electron
        out.append(((1-U)*Tp, om[ok,2], w, np.ones(ok.sum(),bool), tag)) # positron
    return out
'''
open('/tmp/work/mcmod.py','w').write(mcmod)
import importlib, mcmod; importlib.reload(mcmod)
from mcmod import *
print(Phi0_src, Omega)
rng=np.random.default_rng(0)
Ep=np.full(200000,1.25); om=np.tile([0,0,1.0],(200000,1))
eps,cs=sample_kn(Ep,rng); print("mean eps", eps.mean(), "KN check <eps>", )
print(mu_tr_c(1.25), mu_en_c(1.25), mu_tr_c(1.0), mu_en_c(0.5))
