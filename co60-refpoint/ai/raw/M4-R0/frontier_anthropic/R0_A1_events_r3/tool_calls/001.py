
import math, random, time, json
import numpy as np

MEC2 = 0.51099895
RE = 2.8179403e-13          # cm
NE_WATER = 3.3428e23        # electrons per gram, water (rho=1)
ECUT = 0.10                 # MeV, photons below are absorbed (not tracked)

def sigma_KN(E):
    k = E/MEC2
    t1 = (1+k)/k**2*(2*(1+k)/(1+2*k) - math.log(1+2*k)/k)
    t2 = math.log(1+2*k)/(2*k)
    t3 = -(1+3*k)/(1+2*k)**2
    return 2*math.pi*RE**2*(t1+t2+t3)

def mu_C(E):  return NE_WATER*sigma_KN(E)
def mu_R(E):  return 3.0e-4*(0.5/E)**2          # coherent, approx (cm^2/g)
def mu_PE(E): return 3.0e-6*(1.17/E)**3         # photoelectric, negligible
def kappa(E): return 6.5e-4*(E-2*MEC2) if E > 2*MEC2 else 0.0

def mu_parts(E):
    return mu_C(E), mu_R(E), mu_PE(E), kappa(E)

for E in [0.511, 1.1732, 1.3325]:
    c = mu_parts(E); print(E, [round(x,6) for x in c], sum(c))

rnd = random.random
def sample_compton(E):
    k = E/MEC2
    e0 = 1/(1+2*k)
    le0 = math.log(e0)
    while True:
        eps = math.exp(le0*rnd())
        cost = 1 - (1-eps)/(k*eps)
        s2 = 1-cost*cost
        if rnd()*2 < 1+eps*eps-eps*s2:
            return eps*E, cost

def orthobasis(u):
    ux,uy,uz = u
    if abs(ux) < 0.9:
        a = (0.0, -uz, uy)   # cross with x
    else:
        a = (uz, 0.0, -ux)   # cross with y
    n = math.sqrt(a[0]**2+a[1]**2+a[2]**2)
    e1 = (a[0]/n, a[1]/n, a[2]/n)
    e2 = (uy*e1[2]-uz*e1[1], uz*e1[0]-ux*e1[2], ux*e1[1]-uy*e1[0])
    return e1, e2

def rotate(u, cost, phi):
    sint = math.sqrt(max(0.0, 1-cost*cost))
    e1,e2 = orthobasis(u)
    c, s = math.cos(phi), math.sin(phi)
    return tuple(cost*u[i] + sint*(c*e1[i]+s*e2[i]) for i in range(3))

def sample_rayleigh_cos():
    while True:
        c = 2*rnd()-1
        if rnd()*2 < 1+c*c:
            return c

def pe_electron_cos(E):
    beta = math.sqrt(1-(MEC2/(E+MEC2))**2)
    # rejection on sin^2/(1-b c)^4, bound by grid max
    cs = np.linspace(-1,1,2001)
    f = (1-cs**2)/(1-beta*cs)**4
    M = f.max()*1.05
    while True:
        c = 2*rnd()-1
        if rnd()*M < (1-c*c)/(1-beta*c)**4:
            return c

# ---- V geometry: cylinder axis along z, radius 1, z in [4.9,5.1]
ZV0, ZV1, RV = 4.9, 5.1, 1.0
def V_interval(p, u, d):
    x,y,z = p; ux,uy,uz = u
    if uz != 0:
        ta = (ZV0-z)/uz; tb = (ZV1-z)/uz
        lo, hi = min(ta,tb), max(ta,tb)
    else:
        if ZV0 <= z <= ZV1: lo, hi = -1e30, 1e30
        else: return None
    a = ux*ux+uy*uy
    if a < 1e-14:
        if x*x+y*y <= RV*RV: ca, cb = -1e30, 1e30
        else: return None
    else:
        b = 2*(x*ux+y*uy); c = x*x+y*y-RV*RV
        disc = b*b-4*a*c
        if disc <= 0: return None
        sq = math.sqrt(disc)
        ca, cb = (-b-sq)/(2*a), (-b+sq)/(2*a)
    t1 = max(lo, ca, 0.0); t2 = min(hi, cb, d)
    if t2 > t1: return t1, t2
    return None

def exit_dist(p, u):
    x,y,z = p; ux,uy,uz = u
    d = 1e30
    if ux > 0: d = min(d, (15-x)/ux)
    elif ux < 0: d = min(d, (-15-x)/ux)
    if uy > 0: d = min(d, (15-y)/uy)
    elif uy < 0: d = min(d, (-15-y)/uy)
    if uz > 0: d = min(d, (30-z)/uz)
    elif uz < 0: d = min(d, (0-z)/uz)
    return max(d, 0.0)
print("ok")
