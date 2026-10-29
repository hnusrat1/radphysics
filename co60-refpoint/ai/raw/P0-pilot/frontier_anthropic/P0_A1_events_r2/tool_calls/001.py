
import math, random, json, time
import numpy as np

random.seed(12345)
M_E = 0.510998950  # MeV
RE = 2.8179403e-13  # cm
NE = 3.3428e23      # electrons per gram water (rho=1)

def sigma_KN_per_e(E):
    a = E / M_E
    t1 = (1+a)/a**2 * (2*(1+a)/(1+2*a) - math.log(1+2*a)/a)
    t2 = math.log(1+2*a)/(2*a)
    t3 = (1+3*a)/(1+2*a)**2
    return 2*math.pi*RE**2*(t1+t2-t3)

def mu_compton(E):
    return sigma_KN_per_e(E)*NE

def mu_pair(E):
    if E < 2*M_E: return 0.0
    return 0.0022*min(1.0,(E-2*M_E)/(2.0-2*M_E))

MU_PHOTO = 1.5e-5
MU_RAYL = 5e-5

def mu_total(E):
    return mu_compton(E) + mu_pair(E) + MU_PHOTO + MU_RAYL

# Check total at 2 MeV
print("mu_compton(2MeV)=", mu_compton(2.0), " total=", mu_total(2.0))

def kn_sample_eps(E):
    a = E / M_E
    eps0 = 1/(1+2*a); eps0sq = eps0*eps0
    alpha1 = -math.log(eps0)
    alpha2 = alpha1 + 0.5*(1-eps0sq)
    while True:
        if alpha1 > alpha2*random.random():
            eps = math.exp(-alpha1*random.random()); eps2 = eps*eps
        else:
            eps2 = eps0sq + (1-eps0sq)*random.random(); eps = math.sqrt(eps2)
        onecost = (1-eps)/(eps*a)
        sint2 = onecost*(2-onecost)
        greject = 1 - eps*sint2/(1+eps2)
        if greject <= random.random():
            return eps, 1-onecost

def rotate(d, cost, phi):
    # d unit vector; return new unit vector with polar angle acos(cost) and azimuth phi relative to d
    ux,uy,uz = d
    sint = math.sqrt(max(0.0,1-cost*cost))
    if abs(uz) > 0.99999:
        nx = sint*math.cos(phi); ny = sint*math.sin(phi); nz = cost*math.sign(uz)
    else:
        temp = math.sqrt(1-uz*uz)
        nx = sint*(ux*uz*math.cos(phi) - uy*math.sin(phi))/temp + ux*cost
        ny = sint*(uy*uz*math.cos(phi) + ux*math.sin(phi))/temp + uy*cost
        nz = -sint*math.cos(phi)*temp + uz*cost
    n = math.sqrt(nx*nx+ny*ny+nz*nz)
    return (nx/n, ny/n, nz/n)

def in_volume(p):
    x,y,z = p
    return (x*x+y*y <= 1.0) and (9.9 <= z <= 10.1)

def in_phantom(p):
    x,y,z = p
    return abs(x)<15 and abs(y)<15 and 0<z<30

def sample_entry_dir():
    while True:
        u = (2*random.random()-1)*0.05
        v = (2*random.random()-1)*0.05
        w = (1+u*u+v*v)**-1.5
        if random.random() < w:  # uniform in solid angle
            break
    n = math.sqrt(u*u+v*v+1)
    d = (u/n, v/n, 1/n)
    # start at z=-100 (source), entry at z=0: x = u*100/..  position at z=0: t = 100/d_z
    t = 100.0/d[2]
    p = (d[0]*t, d[1]*t, 0.0)
    return p, d

def sample_isotropic_dir():
    c = 2*random.random()-1; ph = 2*math.pi*random.random()
    s = math.sqrt(1-c*c)
    return (s*math.cos(ph), s*math.sin(ph), c)

def track_photon(E, p, d, events, n_hist_stack):
    # stack of photons: (E,p,d)
    stack = [(E,p,d)]
    while stack:
        E,p,d = stack.pop()
        while True:
            mt = mu_total(E)
            s = -math.log(1-random.random())/mt
            # move
            q = (p[0]+d[0]*s, p[1]+d[1]*s, p[2]+d[2]*s)
            if not in_phantom(q):
                break  # leaves phantom (vacuum)
            p = q
            inside = in_volume(p)
            r = random.random()*mt
            mc = mu_compton(E); mp = mu_pair(E)
            if r < mc:
                eps, cost = kn_sample_eps(E)
                Eo = eps*E
                phi = 2*math.pi*random.random()
                pe_x = E - Eo*cost
                pe_mag = math.sqrt(E*E + Eo*Eo - 2*E*Eo*cost)
                cos_e = pe_x/pe_mag if pe_mag>0 else 1.0
                Te = E - Eo
                if inside:
                    events.append(dict(process="compton", e_in_MeV=E, e_out_MeV=Eo, cos_scatter=cost,
                                       t_e_MeV=Te, cos_electron=cos_e, t_positron_MeV=None))
                d = rotate(d, cost, phi)
                E = Eo
                if E < 1e-4:
                    break
            elif r < mc+mp:
                Tt = E - 2*M_E
                fr = random.random()
                Te = Tt*fr; Tp = Tt*(1-fr)
                # electron direction: forward-peaked, angle ~ m/(E_kin+m)
                th = abs(random.gauss(0, M_E/(Te+M_E)))
                cos_e = math.cos(th)
                if inside:
                    events.append(dict(process="pair", e_in_MeV=E, e_out_MeV=None, cos_scatter=None,
                                       t_e_MeV=Te, cos_electron=cos_e, t_positron_MeV=Tp))
                # annihilation at rest: two 0.511 MeV photons back to back, isotropic
                dd = sample_isotropic_dir()
                stack.append((M_E, p, dd))
                stack.append((M_E, p, (-dd[0],-dd[1],-dd[2])))
                break
            elif r < mc+mp+MU_PHOTO:
                if inside:
                    events.append(dict(process="photoelectric", e_in_MeV=E, e_out_MeV=None, cos_scatter=None,
                                       t_e_MeV=E, cos_electron=0.9, t_positron_MeV=None))
                break
            else:
                # Rayleigh: Thomson-like angular distribution (1+c^2) rejection, no energy loss
                while True:
                    c = 2*random.random()-1
                    if random.random()*2 <= 1+c*c: break
                phi = 2*math.pi*random.random()
                if inside:
                    events.append(dict(process="rayleigh", e_in_MeV=E, e_out_MeV=E, cos_scatter=c,
                                       t_e_MeV=0.0, cos_electron=None, t_positron_MeV=None))
                d = rotate(d, c, phi)
    return

events = []
N = 200000
t0 = time.time()
for i in range(N):
    p,d = sample_entry_dir()
    track_photon(2.0, p, d, events, None)
print("histories", N, "events in volume", len(events), "time", time.time()-t0)
from collections import Counter
print(Counter(e['process'] for e in events))
