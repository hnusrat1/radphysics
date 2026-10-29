
import numpy as np, time
rng = np.random.default_rng(2024)
RE=2.8179403e-13; NE=3.3428e23; ME=0.51099891
def sig_KN(E):
    k=E/ME; L=np.log(1+2*k)
    return 2*np.pi*RE**2*((1+k)/k**2*(2*(1+k)/(1+2*k)-L/k)+L/(2*k)-(1+3*k)/(1+2*k)**2)
def mu_c(E): return NE*sig_KN(E)
def mu_R(E): return 0.2*(0.01/E)**1.5
def mu_pe(E): return 4.9*(0.01/E)**3.2
def mu_p(E):
    x=np.clip((E-1.022)/0.978,0,None); return 0.0002*x**2
def mu_tot(E): return mu_c(E)+mu_R(E)+mu_pe(E)+mu_p(E)
print("check mu_c(2)",mu_c(2.0),"mu_tot(2)",mu_tot(2.0),"mu_tot(1)",mu_tot(1.0),"mu_tot(0.1)",mu_tot(0.1))
LO=np.array([-15,-15,0.]); HI=np.array([15,15,30.])
CZ0,CZ1,CR=9.9,10.1,1.0; RB=np.sqrt(1+0.01); CEN=np.array([0,0,10.])

def cyl_chord(o,D):
    a=D[:,0]**2+D[:,1]**2; b=2*(o[:,0]*D[:,0]+o[:,1]*D[:,1]); c=o[:,0]**2+o[:,1]**2-CR**2
    s1=np.full(len(o),-np.inf); s2=np.full(len(o),np.inf); ok=np.ones(len(o),bool)
    par=a<1e-14
    disc=b*b-4*a*c
    ap=~par
    sq=np.sqrt(np.where(disc>=0,disc,0))
    with np.errstate(divide='ignore',invalid='ignore'):
        r1=(-b-sq)/(2*a); r2=(-b+sq)/(2*a)
    lo=np.minimum(r1,r2); hi=np.maximum(r1,r2)
    s1=np.where(ap,np.where(disc>=0,lo,np.inf),np.where(c<=0,-np.inf,np.inf))
    s2=np.where(ap,np.where(disc>=0,hi,-np.inf),np.where(c<=0,np.inf,-np.inf))
    ok&= ~((ap)&(disc<0)) & ~((par)&(c>0))
    Dz=D[:,2]; oz=o[:,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        t1=(CZ0-oz)/Dz; t2=(CZ1-oz)/Dz
    zl=np.where(np.abs(Dz)>1e-14,np.minimum(t1,t2),np.where((oz>=CZ0)&(oz<=CZ1),-np.inf,np.inf))
    zh=np.where(np.abs(Dz)>1e-14,np.maximum(t1,t2),np.where((oz>=CZ0)&(oz<=CZ1),np.inf,-np.inf))
    sin_=np.maximum(s1,zl); sout=np.minimum(s2,zh)
    hit=ok&(sout>sin_)&(sout>0)
    return hit, np.maximum(sin_,0), sout

def exit_dist(p,D):
    with np.errstate(divide='ignore',invalid='ignore'):
        tp=np.where(D>0,(HI-p)/D,np.inf); tm=np.where(D<0,(LO-p)/D,np.inf)
    return np.min(np.minimum(tp,tm),axis=1)

def rotate(axis,c,phi):
    # axis (M,3) unit; c cos, phi azimuth
    a=np.where(np.abs(axis[:,0:1])<0.9,np.array([[1.,0,0]]),np.array([[0,1.,0]]))
    e1=np.cross(axis,a); e1/=np.linalg.norm(e1,axis=1,keepdims=True)
    e2=np.cross(axis,e1)
    s=np.sqrt(np.clip(1-c*c,0,None))
    return c[:,None]*axis+s[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)

def kn_eps(E,rng,n=None):
    # sample Compton eps=E'/E, Kahn-style rejection, vectorized
    k=E/ME; emin=1/(1+2*k); eps=np.empty_like(E)
    todo=np.arange(len(E))
    while len(todo)>0:
        kk=k[todo]; em=emin[todo]
        e=np.exp(rng.uniform(size=len(todo))*np.log(em))
        c=1-(1/e-1)/kk; s2=1-c*c
        acc=rng.uniform(size=len(todo))<=(1+e*e-e*s2)/2
        eps[todo[acc]]=e[acc]
        todo=todo[~acc]
    return eps
print("ok")
