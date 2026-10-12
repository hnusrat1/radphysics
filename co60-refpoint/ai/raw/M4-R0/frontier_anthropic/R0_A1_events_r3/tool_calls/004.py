
code = r'''
import math, random, json, sys
import numpy as np
MEC2 = 0.51099895; RE = 2.8179403e-13; NE_WATER = 3.3428e23
ECUT = 0.10
def sigma_KN(E):
    k = E/MEC2
    t1 = (1+k)/k**2*(2*(1+k)/(1+2*k) - math.log(1+2*k)/k)
    t2 = math.log(1+2*k)/(2*k)
    t3 = -(1+3*k)/(1+2*k)**2
    return 2*math.pi*RE**2*(t1+t2+t3)
def mu_C(E):  return NE_WATER*sigma_KN(E)
def mu_R(E):  return 3.0e-4*(0.5/E)**2
def mu_PE(E): return 3.0e-6*(1.17/E)**3
def kappa(E): return 6.5e-4*(E-2*MEC2) if E > 2*MEC2 else 0.0

rnd = random.random
def sample_compton(E):
    k = E/MEC2; le0 = math.log(1/(1+2*k))
    while True:
        eps = math.exp(le0*rnd())
        cost = 1 - (1-eps)/(k*eps)
        s2 = 1-cost*cost
        if rnd()*2 < 1+eps*eps-eps*s2:
            return eps*E, cost
def orthobasis(u):
    ux,uy,uz = u
    if abs(ux) < 0.9: a = (0.0, -uz, uy)
    else: a = (uz, 0.0, -ux)
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
        if rnd()*2 < 1+c*c: return c
PE_M = {}
def pe_electron_cos(E):
    beta = math.sqrt(1-(MEC2/(E+MEC2))**2)
    cs = np.linspace(-1,1,2001)
    M = ((1-cs**2)/(1-beta*cs)**4).max()*1.05
    while True:
        c = 2*rnd()-1
        if rnd()*M < (1-c*c)/(1-beta*c)**4: return c

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

def make_event(E, W, pos, u, out):
    """Forced/recorded first interaction inside V of photon with energy E."""
    cC, cR, cP, kp = mu_C(E), mu_R(E), mu_PE(E), kappa(E)
    mu = cC+cR+cP+kp
    r = rnd()*mu
    if r < cC:
        Ep, c = sample_compton(E)
        phi = 2*math.pi*rnd()
        Ee = E-Ep
        ce = (E-Ep*c)/math.sqrt(E*E+Ep*Ep-2*E*Ep*c)
        out.append(dict(process="compton", e_in_MeV=E, e_out_MeV=Ep, cos_scatter=c,
                        t_e_MeV=Ee, cos_electron=ce, t_positron_MeV=None, w=W))
    elif r < cC+cR:
        c = sample_rayleigh_cos()
        out.append(dict(process="rayleigh", e_in_MeV=E, e_out_MeV=E, cos_scatter=c,
                        t_e_MeV=0.0, cos_electron=None, t_positron_MeV=None, w=W))
    elif r < cC+cR+cP:
        ce = pe_electron_cos(E)
        out.append(dict(process="photoelectric", e_in_MeV=E, e_out_MeV=None, cos_scatter=None,
                        t_e_MeV=E, cos_electron=ce, t_positron_MeV=None, w=W))
    else:
        Tt = E-2*MEC2; f = rnd()
        th = abs(random.gauss(0,0.6))
        out.append(dict(process="pair", e_in_MeV=E, e_out_MeV=None, cos_scatter=None,
                        t_e_MeV=f*Tt, cos_electron=math.cos(min(th,math.pi)),
                        t_positron_MeV=(1-f)*Tt, w=W))

def run(Nsrc, seed):
    random.seed(seed)
    events = []
    nsrc_total = 0
    for _ in range(Nsrc):
        # source photon
        while True:
            sx = 0.1*rnd()-0.05; sy = 0.1*rnd()-0.05
            wdir = (1+sx*sx+sy*sy)**(-1.5)
            if rnd() < wdir: break   # uniform in solid angle (rejection, weight 1)
        n = math.sqrt(sx*sx+sy*sy+1)
        u = (sx/n, sy/n, 1/n)
        p = (sx*100.0, sy*100.0, 0.0)   # point on front face (source at z=-100)
        E = 1.1732 if rnd() < 0.5 else 1.3325
        stack = [(p, u, E, 1.0/Nsrc)]
        while stack:
            pos, dir_, E, W = stack.pop()
            if E < ECUT: continue
            d = exit_dist(pos, dir_)
            mu = mu_C(E)+mu_R(E)+mu_PE(E)+kappa(E)
            iv = V_interval(pos, dir_, d)
            if iv is None:
                s = -math.log(1-rnd())/mu
                if s > d: continue
                Wc = W
            else:
                t1, t2 = iv
                PV = math.exp(-mu*t1)*(1-math.exp(-mu*(t2-t1)))
                make_event(E, W*PV, pos, dir_, events)
                Pb = 1-math.exp(-mu*t1); Pa = math.exp(-mu*t2)-math.exp(-mu*d)
                Q = Pb+Pa
                if Q <= 0: continue
                v = rnd()
                if rnd()*Q < Pb:
                    s = -math.log(1-v*Pb)/mu
                else:
                    s = -math.log(math.exp(-mu*t2)-v*Pa)/mu
                Wc = W*Q
            # collision at s (not in V)
            x = (pos[0]+s*dir_[0], pos[1]+s*dir_[1], pos[2]+s*dir_[2])
            cC, cR, cP, kp = mu_C(E), mu_R(E), mu_PE(E), kappa(E)
            r = rnd()*mu
            if r < cC:
                Ep, c = sample_compton(E)
                nd = rotate(dir_, c, 2*math.pi*rnd())
                stack.append((x, nd, Ep, Wc))
            elif r < cC+cR:
                c = sample_rayleigh_cos()
                nd = rotate(dir_, c, 2*math.pi*rnd())
                stack.append((x, nd, E, Wc))
            elif r < cC+cR+cP:
                pass
            else:
                # annihilation: two 0.511 MeV photons back to back, isotropic
                ct = 2*rnd()-1; ph = 2*math.pi*rnd()
                st = math.sqrt(1-ct*ct)
                nd = (st*math.cos(ph), st*math.sin(ph), ct)
                stack.append((x, nd, MEC2, Wc))
                stack.append((x, (-nd[0],-nd[1],-nd[2]), MEC2, Wc))
    return events

if __name__ == "__main__":
    Nsrc = int(sys.argv[1]); seed = int(sys.argv[2]); out = sys.argv[3]
    ev = run(Nsrc, seed)
    json.dump(ev, open(out,"w"))
    print(len(ev), sum(e["w"] for e in ev))
'''
open("mc.py","w").write(code)
import subprocess, time
t=time.time()
print(subprocess.run(["python3","mc.py","20000","1","test.json"],capture_output=True,text=True).stdout, time.time()-t)
