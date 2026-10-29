# Debug the photon MC pieces
import numpy as np
from scipy.interpolate import interp1d

ph = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
Eph = ph[:,0]; coh=ph[:,1]; inc=ph[:,2]; pe=ph[:,3]; pn=ph[:,4]; pel=ph[:,5]; tot=ph[:,6]
Elg = np.logspace(np.log10(0.01), np.log10(2.0), 2500)
def interp_log(x, y, xq):
    ok = (y>0)&(x>0)
    fi = interp1d(np.log(x[ok]), np.log(y[ok]), kind='linear', bounds_error=False, fill_value='extrapolate')
    return np.exp(fi(np.log(xq)))
TOT=interp_log(Eph,tot,Elg); logElg=np.log(Elg)
def xs_lookup(arr,E): return np.exp(np.interp(np.log(np.maximum(E,2e-3)),logElg,np.log(arr)))

def cyl_chord(p0, p1, R, zc, hh):
    d = p1-p0
    L = np.sqrt(np.sum(d*d,axis=1))
    dx=d[:,0]; dy=d[:,1]; dz=d[:,2]
    x0=p0[:,0]; y0=p0[:,1]; z0=p0[:,2]
    a = dx*dx+dy*dy
    b = x0*dx+y0*dy
    c = x0*x0+y0*y0-R*R
    big = a>1e-16
    sq = np.sqrt(np.maximum(b*b-a*c,0))
    t1 = (-b-sq)/np.maximum(a,1e-30); t2=(-b+sq)/np.maximum(a,1e-30)
    lo=np.minimum(t1,t2); hi=np.maximum(t1,t2)
    lo[~big]=-1e9; hi[~big]=1e9
    outside = (~big)&(x0*x0+y0*y0>=R*R)
    invdz=np.where(np.abs(dz)>1e-12,1.0/dz,0.0)
    z1=(zc-hh-z0)*invdz; z2=(zc+hh-z0)*invdz
    zlo=np.minimum(z1,z2); zhi=np.maximum(z1,z2)
    dz0=np.abs(dz)<1e-12
    zlo[dz0]=0.; zhi[dz0]=1.
    bad=dz0 & (np.abs(z0-zc)>hh); zlo[bad]=1.; zhi[bad]=0.
    lo=np.maximum(np.maximum(lo,zlo),0); hi=np.minimum(np.minimum(hi,zhi),1)
    chord=L*np.maximum(0,hi-lo)
    chord[outside]=0
    return chord

# test: photon on axis from (0,0,100) to (0,0,110)
p0=np.array([[0.0,0.0,100.0]]); p1=np.array([[0.0,0.0,110.0]])
print("on-axis chord:", cyl_chord(p0,p1,1.0,105.0,0.1))  # expect 0.2
# photon off-axis x=2: from (2,0,100) to (2,0,110)
p0=np.array([[2.0,0.0,100.0]]); p1=np.array([[2.0,0.0,110.0]])
print("off-axis x=2 chord:", cyl_chord(p0,p1,1.0,105.0,0.1))  # expect 0
# photon crossing cylinder diagonally
p0=np.array([[0.5,0.0,100.0]]); p1=np.array([[0.6,0.0,110.0]])
print("diag chord:", cyl_chord(p0,p1,1.0,105.0,0.1))
# sample directions
rng=np.random.default_rng(7)
a=0.05; n=5; u=rng.uniform(-a,a,n); v=rng.uniform(-a,a,n)
d=np.stack([u,v,np.ones(n)],axis=1); d/=np.sqrt(np.sum(d*d,axis=1,keepdims=True))
print("d:",d)
p0=np.stack([u*100,v*100,np.full(n,100.0)],axis=1)
print("p0:",p0)
# mu at 1.17
print("mu_tot:", xs_lookup(TOT,np.array([1.1732]))) 
# first-step test
Ei=np.array([1.1732,1.3325])
mut=xs_lookup(TOT,Ei)
print("mut:",mut)
s_int=-np.log(rng.uniform(1e-300,1,2))/np.maximum(mut,1e-6)
print("s_int:",s_int)
dd=d[:2]; p00=p0[:2]
inv=np.abs(dd)>1e-12
t1=(np.array([-15.,100.,-15.])-p00)/dd; t2=(np.array([15.,130.,15.])-p00)/dd
print("t1:",t1); print("t2:",t2)
tlo=np.maximum(np.minimum(t1,t2),0); thi=np.maximum(t1,t2)
tb=np.min(thi,axis=1)
print("tlo:",tlo,"thi:",thi,"tb:",tb)