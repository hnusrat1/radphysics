
code = r'''
import numpy as np
X=np.loadtxt('/tmp/data/xcom_water.txt')
E_=np.loadtxt('/tmp/data/estar_water.txt')
mc2=0.51099895
re=2.8179403e-13
def loglog(xe, ye, E):
    m = ye>0
    return np.exp(np.interp(np.log(E), np.log(xe[m]), np.log(ye[m])))
XE=X[:,0]
def mu_comp(E):
    coh = loglog(XE,X[:,1],E); inc=loglog(XE,X[:,2],E); pe=loglog(XE,X[:,3],E)
    pn = np.where(E>1.022, loglog(XE,X[:,4],E),0.0)  # pair nuclear (zeros below 1.022 excluded)
    pel= np.where(E>1.022, loglog(XE,np.maximum(X[:,5],1e-30),E),0.0)
    return coh, inc, pe, pn+pel
def mu_all(E):
    coh,inc,pe,pr = mu_comp(E)
    return coh,inc,pe,pr, coh+inc+pe+pr
# ESTAR
EE=E_[:,0]
def estar(col,E):
    return np.exp(np.interp(np.log(E), np.log(EE), np.log(E_[:,col])))
def Scol(E): return estar(1,E)
def Srad(E): return estar(2,E)
def Stot(E): return estar(3,E)
def Yrad(E):
    E=np.asarray(E,float)
    y = np.interp(np.log(E), np.log(EE), E_[:,5])
    return y
def sample_KN(E, rng):
    k = E/mc2
    epsmin = 1/(1+2*k)
    n=len(E)
    eps=np.empty(n); cth=np.empty(n)
    todo=np.arange(n)
    while todo.size:
        kk=k[todo]
        u=rng.random(todo.size)
        ec = epsmin[todo]**u
        c = 1-(1/ec-1)/kk
        s2 = 1-c*c
        p = (ec*ec+1-ec*s2)/2
        acc = rng.random(todo.size) < p
        eps[todo[acc]]=ec[acc]; cth[todo[acc]]=c[acc]
        todo=todo[~acc]
    return eps,cth
def sample_Rayleigh_cos(n,rng):
    out=np.empty(0)
    while out.size<n:
        c=rng.uniform(-1,1,2*n)
        acc = rng.random(2*n) < (1+c*c)/2
        out=np.concatenate([out,c[acc]])
    return out[:n]
def orth_basis(d):
    # d: (n,3) unit
    a=np.where(np.abs(d[:,0:1])<0.9, np.array([[1.,0,0]]), np.array([[0,1.,0]]))
    e1=np.cross(d,a); e1/=np.linalg.norm(e1,axis=1,keepdims=True)
    e2=np.cross(d,e1)
    return e1,e2
def rotate(d,cth,phi):
    s=np.sqrt(np.clip(1-cth*cth,0,1))
    e1,e2=orth_basis(d)
    return cth[:,None]*d + s[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)
'''
open('/tmp/work/phys.py','w').write(code)
import sys; sys.path.insert(0,'/tmp/work')
import importlib, phys; importlib.reload(phys)
rng=np.random.default_rng(1)
E=np.full(200000,2.0)
eps,c=phys.sample_KN(E,rng)
print("KN mean eps at 2MeV", eps.mean(), "mean cos", c.mean())
print(phys.mu_all(np.array([2.0,0.1,0.5])))
print(phys.Yrad(np.array([0.01,1.0,2.0])), phys.Scol(np.array([1.0])))
