import numpy as np, time
# Reuse definitions from previous cell by importing? They are lost. Redefine minimal debug using functions saved? 
# Just rerun a smaller part: replicate photon_mc but print stats on Lcyl for scattered.
# To save time, redefine the function inline quickly is heavy; instead load from the last file? Not possible.
# Quick standalone debug of cyl_seg_length distribution on random scattered-like photons:
xcom = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
E_x = xcom[:,0]; mu_coh=xcom[:,1]; mu_inc=xcom[:,2]; mu_pe=xcom[:,3]
mu_pair=xcom[:,4]+xcom[:,5]; mu_tot=xcom[:,6]
LE = np.log(E_x)
def mut_at(E, mu):
    mu = np.maximum(mu,1e-30)
    return np.exp(np.interp(np.log(np.clip(E,1e-6,20.)), LE, np.log(mu)))

def cyl_seg_length(p0, d, s, z0=4.9, z1=5.1, R=1.0):
    x0=p0[:,0]; y0=p0[:,1]; z0v=p0[:,2]
    dx=d[:,0]; dy=d[:,1]; dz=d[:,2]
    dzs = np.where(dz==0, 1e-12, dz)
    t1=(z0-z0v)/dzs; t2=(z1-z0v)/dzs
    tmin=np.minimum(t1,t2); tmax=np.maximum(t1,t2)
    tin=np.maximum(0.0,tmin); tout=np.minimum(s,tmax)
    a=dx*dx+dy*dy; a=np.maximum(a,1e-16)
    b=2*(x0*dx+y0*dy); c=x0*x0+y0*y0-R*R
    disc=b*b-4*a*c
    ok=disc>0
    sq=np.sqrt(np.where(ok,disc,0.0))
    t1c=(-b-sq)/(2*a); t2c=(-b+sq)/(2*a)
    tinc=np.maximum(np.where(ok,t1c,0.0),0.0)
    toutc=np.minimum(np.where(ok,t2c,s),s)
    tin_=np.maximum(tin,tinc); tout_=np.minimum(tout,toutc)
    L=np.maximum(0.0,tout_-tin_)
    # sanity: if disc<=0 but inside cylinder, we'd miss; check
    return L

rng=np.random.default_rng(0)
# scattered-like: random positions inside cube, random dir, random segment len
N=200000
p0=np.column_stack([rng.uniform(-15,15,N), rng.uniform(-15,15,N), rng.uniform(0,30,N)])
# isotropic dir
cth=rng.uniform(-1,1,N); sth=np.sqrt(1-cth*cth); phi=rng.uniform(0,2*np.pi,N)
d=np.column_stack([sth*np.cos(phi), sth*np.sin(phi), cth])
s=rng.uniform(0,10,N)
L=cyl_seg_length(np.asarray(p0,float), np.asarray(d,float), s)
print('max L', L.max(), 'sum L', L.sum(), 'nonzero frac', (L>0).mean())
# fraction that would be inside: expected small
# check disc<=0 but inside: 
inside=(p0[:,0]**2+p0[:,1]**2)<1
print('inside at t=0 non-cylinder-intercept cases', ((inside)&(L==0)).sum())