
code = r'''
import numpy as np
from scipy import integrate
def load(fn):
    rows=[l.split() for l in open(fn) if l.strip() and not l.startswith('#')]
    return np.array([[float(x) for x in r] for r in rows])
X=load('/tmp/data/xcom_water.txt')
S=load('/tmp/data/estar_water.txt')
Ex=X[:,0]
def lli(x,y,xn):
    return np.exp(np.interp(np.log(xn),np.log(x),np.log(np.maximum(y,1e-300))))
def mu_inc(E): return lli(Ex,X[:,2],E)
def mu_coh(E): return lli(Ex,X[:,1],E)
def mu_pe(E): return lli(Ex,X[:,3],E)
def mu_pair(E): return lli(Ex,X[:,4]+X[:,5],E)
def mutot(E): return lli(Ex,X[:,6],E)
re=2.8179403e-13; ne=3.3428e23; mec2=0.51099895
Om_p=integrate.dblquad(lambda v,u:(1+u*u+v*v)**-1.5, -0.05,0.05, -0.05,0.05)[0]
def kn_f(E,mu):
    k=E/mec2; eps=1/(1+k*(1-mu))
    return eps*eps*(eps+1/eps-(1-mu*mu))
def kn_fmax(E):
    mug=np.linspace(-1,1,401)
    return kn_f(np.asarray(E)[...,None],mug[None,:]).max(axis=-1)*1.001
def kn_sample_mu(E,rng):
    n=len(E); fmax=kn_fmax(E)
    out=np.empty(n); todo=np.arange(n)
    while len(todo):
        mu=rng.uniform(-1,1,len(todo))
        f=kn_f(E[todo],mu)
        acc=rng.uniform(0,1,len(todo))*fmax[todo]<f
        out[todo[acc]]=mu[acc]; todo=todo[~acc]
    return out
def rot(d,mu,phi):
    # d: (n,3) unit; returns unit vectors at angle acos(mu), azimuth phi
    n=len(d)
    a=np.where((np.abs(d[:,2])<0.9)[:,None], np.cross(d,[0,0,1.0]), np.cross(d,[1.0,0,0]))
    a/=np.linalg.norm(a,axis=1)[:,None]
    b=np.cross(d,a)
    s=np.sqrt(np.clip(1-mu*mu,0,1))
    return mu[:,None]*d+s[:,None]*(np.cos(phi)[:,None]*a+np.sin(phi)[:,None]*b)
def iso(n,rng):
    z=rng.uniform(-1,1,n); p=rng.uniform(0,2*np.pi,n); s=np.sqrt(1-z*z)
    return np.stack([s*np.cos(p),s*np.sin(p),z],1)
def ray_exit(pos,d):
    # distance to exit box x,y in +-15, z in [0,30]
    t=np.full(len(pos),np.inf)
    for ax,lo,hi in [(0,-15,15),(1,-15,15),(2,0,30)]:
        dd=d[:,ax]
        with np.errstate(divide='ignore',invalid='ignore'):
            tp=np.where(dd>1e-14,(hi-pos[:,ax])/dd,np.inf)
            tm=np.where(dd<-1e-14,(lo-pos[:,ax])/dd,np.inf)
        t=np.minimum(t,np.minimum(tp,tm))
    return t
'''
open('/tmp/work/lib.py','w').write(code) if __import__('os').path.isdir('/tmp/work') else None
import os
os.makedirs('/tmp/work',exist_ok=True)
open('/tmp/work/lib.py','w').write(code)
print(subprocess.run(['bash','-lc','cd /tmp/work && python -c "import lib; print(lib.Om_p, lib.mutot(1.1732))"'],capture_output=True,text=True))
