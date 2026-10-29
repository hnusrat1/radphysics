import numpy as np, time

# ---------------- data ----------------
xcom = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
E_x = xcom[:,0]; mu_coh=xcom[:,1]; mu_inc=xcom[:,2]; mu_pe=xcom[:,3]
mu_pair=xcom[:,4]+xcom[:,5]; mu_tot=xcom[:,6]
LE = np.log(E_x)
def mut_at(E, mu):
    return np.exp(np.interp(np.log(np.clip(E,1e-6,20.)), LE, np.log(mu)))

m_e = 0.5109989461

def sample_compton(E, rng):
    E = np.asarray(E, dtype=np.float64); n=E.size
    alpha = E/m_e
    K=10
    mu_c = rng.uniform(-1,1,(n,K)); u_c = rng.random((n,K))
    d = 1.0 + alpha[:,None]*(1.0-mu_c)
    p = ((1+mu_c*mu_c)/2.0 + (alpha[:,None]**2*(1-mu_c)**2)/(2.0*d)) / (d*d)
    accept = u_c < p; has = accept.any(axis=1)
    mu = mu_c[np.arange(n), np.argmax(accept,axis=1)]
    bad = ~has; cnt=0
    while bad.any() and cnt<50:
        nb=bad.sum()
        mc=rng.uniform(-1,1,(nb,K)); uc=rng.random((nb,K))
        dd=1.0+alpha[bad][:,None]*(1.0-mc)
        pp=((1+mc*mc)/2.0+(alpha[bad][:,None]**2*(1-mc)**2)/(2.0*dd))/(dd*dd)
        acc=uc<pp; h2=acc.any(axis=1)
        mu[bad]=mc[np.arange(nb), np.argmax(acc,axis=1)]
        bad[bad]=~h2; cnt+=1
    if bad.any(): mu[bad]=rng.uniform(-1,1,bad.sum())
    Ep = E/(1.0+alpha*(1.0-mu))
    pe = np.sqrt(np.maximum(E*E+Ep*Ep-2*E*Ep*mu,0))
    cos_e = (E-Ep*mu)/pe
    return mu, Ep, cos_e, E-Ep

def cyl_seg_length(p0, d, s):
    # p0 (N,3), d (N,3) unit, s (N)>=0. cylinder z[4.9,5.1], r=1
    x0=p0[:,0]; y0=p0[:,1]; z0=p0[:,2]
    dx=d[:,0]; dy=d[:,1]; dz=d[:,2]
    eps=1e-12
    t1=(4.9-z0)/(dz+eps*(dz==0)); t2=(5.1-z0)/(dz+eps*(dz==0))
    tmin=np.minimum(t1,t2); tmax=np.maximum(t1,t2)
    tin=np.maximum(0.0,tmin); tout=np.minimum(s,tmax)
    a=dx*dx+dy*dy
    a=np.maximum(a,1e-16)
    b=2*(x0*dx+y0*dy); c=x0*x0+y0*y0-1.0
    disc=b*b-4*a*c
    ok=disc>0
    sq=np.sqrt(np.where(ok,disc,0.0))
    t1c=(-b-sq)/(2*a); t2c=(-b+sq)/(2*a)
    tinc=max... 
    return None