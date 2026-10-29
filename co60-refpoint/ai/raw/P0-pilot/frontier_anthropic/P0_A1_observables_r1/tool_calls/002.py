
import numpy as np, time
rng=np.random.default_rng(1)
mc2=0.51099895
def mu_pe(E): return 3.0e-6*E**-3
def mu_coh(E): return 0.005*(0.1/E)**1.07
def mu_pair(E): return np.where(E>1.022, 0.0004*np.clip((E-1.022)/0.5,0,1),0.0)
def mu_c(E): return ne*sig_KN(E)
def mu_tot(E): return mu_c(E)+mu_pe(E)+mu_pair(E)+mu_coh(E)
# mean scattered energy fraction <r>(E)
cg=np.linspace(-1,1,20001)
Egrid=np.logspace(-3,np.log10(2.5),400)
rmean=[]
for e in Egrid:
    k=e/mc2; r=1/(1+k*(1-cg)); f=r**2*(r+1/r-(1-cg**2))
    rmean.append(np.trapz(f*r,cg)/np.trapz(f,cg))
rmean=np.array(rmean)
def Tbar(E): return E*(1-np.interp(np.log(E),np.log(Egrid),rmean))
def sample_KN(E,n=None):
    E=np.asarray(E,float); out_c=np.empty(E.shape); 
    todo=np.arange(E.size)
    Ec=E.copy()
    while todo.size:
        k=Ec[todo]/mc2
        c=rng.uniform(-1,1,todo.size)
        r=1/(1+k*(1-c))
        f=r**2*(r+1/r-(1-c**2))
        acc=rng.uniform(0,2,todo.size)<f
        out_c[todo[acc]]=c[acc]
        todo=todo[~acc]
    r=1/(1+E/mc2*(1-out_c))
    Ep=E*r
    return Ep,out_c
def rotate(d,c,phi):
    # d: (n,3) unit; returns unit vector at angle acos(c), azimuth phi
    n=d.shape[0]
    a=np.where(np.abs(d[:,[0]])<0.9,np.array([[1,0,0]]),np.array([[0,1,0]]))
    u=np.cross(d,a); u/=np.linalg.norm(u,axis=1)[:,None]
    v=np.cross(d,u)
    s=np.sqrt(np.clip(1-c**2,0,1))
    return c[:,None]*d+s[:,None]*(np.cos(phi)[:,None]*u+np.sin(phi)[:,None]*v), u, v
# check Tbar and KN sampler mean
Ep,cc=sample_KN(np.full(200000,2.0))
print("mean Ep/E sampled",Ep.mean()/2, "table", rmean[np.argmin(abs(Egrid-2.0))])
print("Tbar/E at 2MeV", Tbar(2.0)/2.0)
