import numpy as np, math, time, sys
from collections import Counter

rng = np.random.default_rng(12345)

me = 0.51099895
re = 2.8179403262e-13
Ne = 3.343e23

def kn_f(mu, E):
    a = E/me
    r = 1.0/(1.0+a*(1.0-mu))
    return r*r*(1.0/r + r - (1.0-mu*mu))

Egrid = np.logspace(np.log10(0.001), np.log10(10.0), 300)
sig_grid = np.empty(300)
for j,E in enumerate(Egrid):
    mus = np.linspace(-1,1,3001)
    sig_grid[j] = np.pi*re**2 * np.trapezoid(kn_f(mus,E), mus)

E_ph = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.15,0.20,0.30,0.50,0.80,1.0,1.5,2.0])
mu_ph= np.array([4.90,0.506,0.162,0.0712,0.0375,0.0223,0.00988,0.00547,0.00240,0.00128,0.000526,0.000151,5.0e-5,3.0e-5,1.0e-5,4.7e-6])
E_pa = np.array([1.022,1.25,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0,10.0])
mu_pa= np.array([0.0,3.1e-5,1.32e-4,4.1e-4,8.1e-4,1.39e-3,2.69e-3,4.32e-3,6.19e-3,1.05e-2,1.55e-2])
E_co = np.array([0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20,0.30,0.50,0.80,1.0,1.5,2.0])
mu_co= np.array([0.30,0.060,0.022,0.0098,0.0046,0.0034,0.0017,0.0011,0.0006,0.0003,0.00015,0.00012,8e-5,4.5e-5])

def mu_comp(E):
    logE = np.log(np.clip(E,1e-3,10))
    return Ne * np.interp(logE, np.log(Egrid), sig_grid)
def mu_photo(E):
    logE = np.log(np.clip(E,E_ph[0],E_ph[-1]))
    return np.exp(np.interp(logE, np.log(E_ph), np.log(mu_ph)))
def mu_pair(E):
    res = np.zeros_like(E)
    mask = E>1.022
    if mask.any():
        logE = np.log(np.clip(E[mask],E_pa[1],E_pa[-1]))
        res[mask] = np.exp(np.interp(logE, np.log(E_pa[1:]), np.log(mu_pa[1:])))
    return res
def mu_coh(E):
    logE = np.log(np.clip(E,E_co[0],E_co[-1]))
    return np.exp(np.interp(logE, np.log(E_co), np.log(mu_co)))

def sample_compton(E, rng):
    n = len(E)
    mu = np.empty(n)
    alive = np.ones(n,bool)
    a = E/me
    while alive.any():
        idx = np.where(alive)[0]
        mu_try = rng.uniform(-1,1,len(idx))
        f = kn_f(mu_try, E[idx])
        acc = rng.uniform(0,2,len(idx)) < f
        mu[idx[acc]] = mu_try[acc]
        alive[idx[acc]] = False
    Ep = E / (1.0 + a*(1.0-mu)
    te = E - Ep
    num = E - Ep*mu
    den = np.sqrt(E*E + Ep*Ep - 2.0*E*Ep*mu)
    cos_el = np.where(den>1e-15, num/den, 1.0)
    return mu, Ep, te, cos_el

def rotate_dir(ux,uy,uz, mu_s, phi):
    d = np.column_stack([ux,uy,uz])
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    ref = np.tile(np.array([0.,0.,1.]),(len(d),1))
    nz = np.abs(d[:,2]) > 0.9
    ref[nz] = [1.,0.,0.]
    e1 = np.cross(d,ref)
    e1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-30)
    e2 = np.cross(d,e1)
    st = np.sqrt(np.clip(1.0-mu_s*mu_s,0,1))
    nd = (st*np.cos(phi))[:,np.newaxis]*e1 + (st*np.sin(phi))[:,np.newaxis]*e2 + mu_s[:,np.newaxis]*d
    nd = nd / np.linalg.norm(nd, axis=1, keepdims=True)
    return nd[:,0], nd[:,1], nd[:,2]

def inside_cube(x,y,z):
    return (np.abs(x)<15)&(np.abs(y)<15)&(z>100)&(z<130)
def in_scoring(x,y,z):
    return (x*x+y*y<1.0)&(z>109.9)&(z<110.1)

print("All functions ready")