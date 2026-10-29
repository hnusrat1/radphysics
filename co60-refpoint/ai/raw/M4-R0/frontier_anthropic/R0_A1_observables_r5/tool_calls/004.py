
code = r'''
import numpy as np
from phys import *
Omega = 4*np.arcsin(np.sin(np.arctan(0.05))**2)   # solid angle of pyramid (sr)
D_SRC=100.0; R_SV=1.0; Z1,Z2=4.9,5.1; ZC=5.0
def mu_pe(E): return 0.0028*(0.1/E)**3
def mu_coh(E): return 0.0045*(0.1/E)**2
def mu_pair(E):
    x=np.clip((E-1.022)/0.3,0,None)
    return np.where(E>1.022, 2e-4*x**2, 0.0)
def mu_c(E): return ne_w*kn_sigma(E)
def sample_KN(E,rng):
    k=E/me
    n=E.shape[0]
    eps=np.empty(n); todo=np.arange(n)
    emin=1/(1+2*k)
    while todo.size:
        kk=k[todo]; em=emin[todo]
        r1=rng.random(todo.size); r2=rng.random(todo.size)
        e=em**r1          # pdf ~1/eps on [emin,1]
        f=e+1/e-(1-(1-(1/e-1)/kk)**2)
        acc=r2*2 < f*e
        eps[todo[acc]]=e[acc]
        todo=todo[~acc]
    c=1-(1/eps-1)/k
    return eps,c
def rotate(u,c,phi):
    # u: (n,3) unit; return new unit vectors at polar c, azimuth phi
    n=u.shape[0]
    s=np.sqrt(np.clip(1-c*c,0,None))
    # build orthonormal basis
    a=np.where(np.abs(u[:,0:1])<0.9, np.array([[1.,0,0]]), np.array([[0,1.,0]]))
    e1=np.cross(u,a); e1/=np.linalg.norm(e1,axis=1,keepdims=True)
    e2=np.cross(u,e1)
    return (c[:,None]*u + s[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2))
def sample_thomson(n,rng):
    # rejection sampling of (1+c^2)
    out=np.empty(n); todo=np.arange(n)
    while todo.size:
        c=rng.uniform(-1,1,todo.size); r=rng.random(todo.size)
        acc=r*2<(1+c*c)
        out[todo[acc]]=c[acc]; todo=todo[~acc]
    return out
def exit_distance(p,u):
    # cube x,y in [-15,15], z in [0,30]
    with np.errstate(divide='ignore',invalid='ignore'):
        tx=np.where(u[:,0]>0,(15-p[:,0])/u[:,0],np.where(u[:,0]<0,(-15-p[:,0])/u[:,0],np.inf))
        ty=np.where(u[:,1]>0,(15-p[:,1])/u[:,1],np.where(u[:,1]<0,(-15-p[:,1])/u[:,1],np.inf))
        tz=np.where(u[:,2]>0,(30-p[:,2])/u[:,2],np.where(u[:,2]<0,(0-p[:,2])/u[:,2],np.inf))
    return np.minimum(np.minimum(tx,ty),tz)
def sv_track(p,u,step,z1,z2,R=R_SV):
    # track length of segment p + t u, t in [0,step], inside cylinder radius R, z in [z1,z2]
    a=u[:,0]**2+u[:,1]**2
    b=2*(p[:,0]*u[:,0]+p[:,1]*u[:,1])
    c=p[:,0]**2+p[:,1]**2-R*R
    lo=np.zeros(len(step)); hi=step.copy()
    # radial
    disc=b*b-4*a*c
    with np.errstate(divide='ignore',invalid='ignore'):
        sq=np.sqrt(np.clip(disc,0,None))
        r1=(-b-sq)/(2*a); r2=(-b+sq)/(2*a)
    inside_all=(c<=0)
    rad_lo=np.where(a>1e-14, np.where(disc>=0, np.minimum(r1,r2), np.inf), np.where(inside_all,-np.inf,np.inf))
    rad_hi=np.where(a>1e-14, np.where(disc>=0, np.maximum(r1,r2), -np.inf), np.where(inside_all,np.inf,-np.inf))
    lo=np.maximum(lo,rad_lo); hi=np.minimum(hi,rad_hi)
    with np.errstate(divide='ignore',invalid='ignore'):
        za=(z1-p[:,2])/u[:,2]; zb=(z2-p[:,2])/u[:,2]
    zlo=np.where(u[:,2]!=0,np.minimum(za,zb),np.where((p[:,2]>=z1)&(p[:,2]<=z2),-np.inf,np.inf))
    zhi=np.where(u[:,2]!=0,np.maximum(za,zb),np.where((p[:,2]>=z1)&(p[:,2]<=z2),np.inf,-np.inf))
    lo=np.maximum(lo,zlo); hi=np.minimum(hi,zhi)
    return np.clip(hi-lo,0,None), lo
'''
open('mc.py','w').write(code)
import importlib
import mc
print(mc.Omega)
