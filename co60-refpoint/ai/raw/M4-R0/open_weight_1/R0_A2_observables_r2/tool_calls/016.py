import numpy as np

def load_xcom():
    with open('/tmp/data/xcom_water.txt') as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    x = np.loadtxt(lines)
    return x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5], x[:,6]
E_x, mu_coh, mu_incoh, mu_pe, mu_pairn, mu_paire, mu_tot = load_xcom()
def mu_interp(E, col):
    return np.exp(np.interp(np.log(E), np.log(E_x), np.log(col)))
def mu_total(E): return mu_interp(E, mu_tot)

def cyl_intersect(xa, ua, z0, z1):
    aq=ua[:,0]**2+ua[:,1]**2; bq=2.0*(xa[:,0]*ua[:,0]+xa[:,1]*ua[:,1]); cq=xa[:,0]**2+xa[:,1]**2-1.0
    par=aq<1e-14; tin=np.full(aq.shape,-np.inf); tout=np.full(aq.shape,np.inf); ok=np.ones(aq.shape,bool)
    npar=~par; disc=bq[npar]**2-4*aq[npar]*cq[npar]; pos=disc>=0; sq=np.sqrt(np.maximum(disc,0))
    t1=(-bq[npar]-sq)/(2*aq[npar]); t2=(-bq[npar]+sq)/(2*aq[npar]); ti=np.minimum(t1,t2); to=np.maximum(t1,t2)
    tin_np=tin[npar]; tout_np=tout[npar]; tin_np[~pos]=np.inf; tout_np[~pos]=-np.inf
    tin_np[pos]=ti[pos]; tout_np[pos]=to[pos]; tin[npar]=tin_np; tout[npar]=tout_np
    parrad=(xa[par,0]**2+xa[par,1]**2)<=1.0; ok[par]=parrad
    with np.errstate(divide='ignore',invalid='ignore'):
        tza=np.where(ua[:,2]>0,(z0-xa[:,2])/ua[:,2],np.where(ua[:,2]<0,(z0-xa[:,2])/ua[:,2],np.inf))
        tzb=np.where(ua[:,2]>0,(z1-xa[:,2])/ua[:,2],np.where(ua[:,2]<0,(z1-xa[:,2])/ua[:,2],np.inf))
    tzlo=np.minimum(tza,tzb); tzhi=np.maximum(tza,tzb)
    tlo=np.maximum(tin,tzlo); thi=np.minimum(tout,tzhi)
    valid=ok&(thi>tlo)&(thi>0)
    return tlo,thi,valid

rng=np.random.default_rng(3); n=100000
E=np.where(rng.random(n)<0.5,1.1732,1.3325)
xs=rng.uniform(-5,5,n); ys=rng.uniform(-5,5,n)
d=np.stack([xs,ys,np.full(n,100.0)],axis=1); nrm=np.linalg.norm(d,axis=1); u=d/nrm[:,None]
x=np.stack([xs,ys,np.zeros(n)],axis=1)
prim=np.ones(n,bool); alive=np.ones(n,bool); mu=mu_total(E)
idx=np.where(alive)[0]; m=idx.size
Ea=E[idx]; ua=u[idx]; xa=x[idx]; mua=mu[idx]; pra=prim[idx]
print("pra sum:", pra.sum(), "pra dtype", pra.dtype)
print("prim[alive] sum:", prim[alive].sum(), "alive sum:", alive.sum())
pr=pra.astype(int)
print("pr unique:", np.unique(pr))
tlo,thi,valid=cyl_intersect(xa,ua,4.9,5.1)
t_in=np.maximum(tlo,0.0); L=np.maximum(thi-t_in,0.0)
contrib=np.where(valid,np.exp(-mua*t_in)*L/(np.pi*0.2),0.0)
score=np.zeros((28,2,10))
ph_edges=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
ct_edges=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
Eb=np.clip(np.searchsorted(ph_edges,Ea,side='right')-1,0,27)
cb=np.clip(np.searchsorted(ct_edges,ua[:,2],side='right')-1,0,9)
np.add.at(score,(Eb,pr,cb),contrib)
print("primary total:",score[:,0,:].sum(),"scattered total:",score[:,1,:].sum())