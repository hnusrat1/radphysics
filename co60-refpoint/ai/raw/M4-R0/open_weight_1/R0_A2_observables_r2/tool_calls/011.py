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
    aq = ua[:,0]**2 + ua[:,1]**2
    bq = 2.0*(xa[:,0]*ua[:,0] + xa[:,1]*ua[:,1])
    cq = xa[:,0]**2 + xa[:,1]**2 - 1.0
    par = aq < 1e-14
    tin = np.full(aq.shape, -np.inf); tout = np.full(aq.shape, np.inf)
    ok = np.ones(aq.shape, bool)
    npar = ~par
    disc = bq[npar]**2 - 4*aq[npar]*cq[npar]
    pos = disc >= 0
    sq = np.sqrt(np.maximum(disc,0))
    t1 = (-bq[npar] - sq)/(2*aq[npar]); t2 = (-bq[npar] + sq)/(2*aq[npar])
    ti = np.minimum(t1,t2); to = np.maximum(t1,t2)
    tin_np = tin[npar]; tout_np = tout[npar]
    tin_np[~pos] = np.inf; tout_np[~pos] = -np.inf
    tin_np[pos] = ti[pos]; tout_np[pos] = to[pos]
    tin[npar]=tin_np; tout[npar]=tout_np
    parrad = (xa[par,0]**2 + xa[par,1]**2) <= 1.0
    ok[par] = parrad
    with np.errstate(divide='ignore',invalid='ignore'):
        tza = np.where(ua[:,2]>0, (z0-xa[:,2])/ua[:,2], np.where(ua[:,2]<0,(z0-xa[:,2])/ua[:,2], np.inf))
        tzb = np.where(ua[:,2]>0, (z1-xa[:,2])/ua[:,2], np.where(ua[:,2]<0,(z1-xa[:,2])/ua[:,2], np.inf))
    tzlo = np.minimum(tza,tzb); tzhi = np.maximum(tza,tzb)
    tlo = np.maximum(tin, tzlo); thi = np.minimum(tout, tzhi)
    valid = ok & (thi>tlo) & (thi>0)
    return tlo, thi, valid

rng = np.random.default_rng(42)
n=2000000
E = np.where(rng.random(n)<0.5, 1.1732, 1.3325)
xs = rng.uniform(-5,5,n); ys = rng.uniform(-5,5,n)
d = np.stack([xs, ys, np.full(n,100.0)],axis=1)
nrm = np.linalg.norm(d,axis=1)
u = d/nrm[:,None]
x = np.stack([xs, ys, np.zeros(n)],axis=1)
mu = mu_total(E)
tlo, thi, valid = cyl_intersect(x, u, 4.9, 5.1)
t_in = np.maximum(tlo,0.0)
L = np.maximum(thi-t_in,0.0)
contrib = valid * np.exp(-mu*t_in)*L/(np.pi*0.2)
per_photon = contrib.sum()/n
print("per-photon primary track estimate (cm^-2):", per_photon)
print("times Omega r^2 (110.25):", per_photon*110.25)
# analytic: fraction aimed within cylinder x integral
# compute analytic via average exp(-mu*path) over surface points within mapping
sel = valid
print("fraction valid:", valid.mean())
print("avg exp(-mu t_in) among valid:", np.exp(-mu[sel]*t_in[sel]).mean())
print("avg L among valid:", L[sel].mean())
# analytic check for the axial line
for E0 in [1.1732,1.3325]:
    m = mu_total(np.array([E0]))[0]
    ana = (np.exp(-m*4.9)-np.exp(-m*5.1))/(m*0.2)*0.2/(0.2)  # not quite; do surface integral
    print(E0, "mu",m)
# compute surface integral: fraction(e^{-mu path} L/V) analytically over the 10x10 square
# path = z/uz where z=4.9..5.1 integrated
import scipy.integrate as si
def analytic(E0):
    mu = mu_total(np.array([E0]))[0]
    # integral over surface of exp(-mu*path(zavg))*0.2/V * I(within cyl mapping), average over 100 cm^2
    # For each surface point (xs,ys), cylinder at entry z=4.9: lateral scaling f=104.9/100; check r_surf<=1/f
    # path from z=0 to 5.1: use average: integral in z.
    # contribution = pi r_c^2/100 * (1/0.2) integral_{4.9}^{5.1} exp(-mu*z/uz) dz /V? careful
    # Actually track-length per volume: (1/V)*exp(-mu t) dt integrated over z -> (1/V)*(1/mu/uz)*(e^{-mu*4.9/uz}-e^{-mu*5.1/uz})
    # For axial uz=1.
    rc = 1.0/(104.9/100.0)  # surface radius mapping to radius-1 cylinder at z=4.9 entry
    frac = np.pi*rc**2/100.0
    # integrate z from 4.9 to 5.1 with attenuation:
    zz = np.linspace(4.9,5.1,10001)
    # for each photon the path in water to depth z is z/uz; uz~1 for valid ones (near axis), but for those within cylinder mapping at 4.9, uz varies slightly
    # approximate uz=1
    trk = np.trapezoid(np.exp(-mu*zz), zz)  # = (1/mu)(e^-mu4.9 - e^-mu5.1)
    contrib_avg = frac * trk / (0.2* np.pi)  # track length per photon: frac * trk ; per volume: /V where V=pi*0.2
    return contrib_avg*110.25
for E0 in [1.1732,1.3325]:
    print("analytic primary/Phi0 for line",E0, analytic(E0))
print("mean of two lines:", 0.5*(analytic(1.1732)+analytic(1.3325)))