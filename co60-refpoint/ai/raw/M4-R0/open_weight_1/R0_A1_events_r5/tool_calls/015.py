import numpy as np
from scipy.interpolate import interp1d
import json

np.random.seed(42)

# ============ CONSTANTS ============
m_e = 0.51099895
r_e = 2.8179403262e-13
NA = 6.02214076e23

# ============ CROSS SECTIONS ============
ea = np.array([1.0, 1.1, 1.1732, 1.2, 1.25, 1.3, 1.3325, 1.4, 1.5])
mt = np.array([0.07070, 0.06780, 0.06590, 0.06520, 0.06400, 0.06280, 0.06220, 0.06080, 0.05780])
mc = np.array([0.06950, 0.06650, 0.06430, 0.06350, 0.06200, 0.06060, 0.06000, 0.05850, 0.05550])
mph = np.array([0.00040, 0.00030, 0.00025, 0.00022, 0.00020, 0.00017, 0.00015, 0.00013, 0.00010])
mpp = np.array([0.0, 0.0, 0.00020, 0.00028, 0.00050, 0.00080, 0.00100, 0.00130, 0.00150])
mrl = np.array([0.00080, 0.00068, 0.00060, 0.00055, 0.00050, 0.00043, 0.00040, 0.00035, 0.00030])

ft = interp1d(ea, mt, kind='cubic', bounds_error=False, fill_value='extrapolate')
fc = interp1d(ea, mc, kind='cubic', bounds_error=False, fill_value='extrapolate')
fph = interp1d(ea, mph, kind='cubic', bounds_error=False, fill_value='extrapolate')
fpp = interp1d(ea, mpp, kind='cubic', bounds_error=False, fill_value='extrapolate')
frl = interp1d(ea, mrl, kind='cubic', bounds_error=False, fill_value='extrapolate')

def mu_tot(E): return float(ft(np.clip(E, 0.01, 10.0)))
def mu_comp(E): return float(fc(np.clip(E, 0.01, 10.0)))
def mu_phot(E): return float(fph(np.clip(E, 0.01, 10.0)))
def mu_pair(E): return float(fpp(np.clip(E, 0.01, 10.0)))
def mu_rayl(E): return float(frl(np.clip(E, 0.01, 10.0)))

# ============ GEOMETRY ============
SSD = 100.0
src = np.array([0.0, 0.0, -SSD])
fh = 5.0  # field half-width
zs = 0.0  # surface z
zc = 5.0  # scoring center
zh = 0.1  # half-thickness
zlo = zc - zh
zhi = zc + zh
Rc = 1.0  # cylinder radius
ctr = np.array([0.0, 0.0, zc])

th = np.arctan(fh / SSD)
Opyr = 4 * np.arcsin(np.sin(th)**2)
ne = 10.0 * NA / 18.015

print(f"Opyr = {Opyr:.6f} sr")
print(f"ne = {ne:.4e}")

# ============ PHYSICS ============
def c_E(E, ct): return E / (1.0 + (E/m_e)*(1.0 - ct))

def KN(E, ct):
    th = np.arccos(np.clip(ct, -1.0, 1.0))
    Eo = c_E(E, ct)
    return (r_e**2/2.0) * (Eo/E)**2 * (E/Eo + Eo/E - np.sin(th)**2)

def sample_KN(E):
    mx = r_e**2 * 1.05
    while True:
        ct = np.random.uniform(-1.0, 1.0)
        if np.random.random() * mx < KN(E, ct):
            break
    Eo = c_E(E, ct)
    te = E - Eo
    num = E - Eo * ct
    den = np.sqrt(E**2 + Eo**2 - 2*E*Eo*ct)
    ce = num/den if den > 0 else 1.0
    return Eo, ct, te, ce

def rot(v, ct):
    u = v / np.linalg.norm(v)
    th = np.arccos(np.clip(ct, -1.0, 1.0))
    ph = np.random.uniform(0.0, 2.0*np.pi)
    if abs(u[2]) < 0.999:
        w = np.cross(u, np.array([0.0, 0.0, 1.0]))
    else:
        w = np.cross(u, np.array([1.0, 0.0, 0.0]))
    w = w / np.linalg.norm(w)
    vv = np.cross(w, u)
    return u*ct + w*np.sin(th)*np.cos(ph) + vv*np.sin(th)*np.sin(ph)

def src_samp():
    E = 1.1732 if np.random.random() < 0.5 else 1.3325
    x = np.random.uniform(-fh, fh)
    y = np.random.uniform(-fh, fh)
    d = np.array([x, y, SSD])
    d = d / np.linalg.norm(d)
    return E, d, x, y

def proc(E):
    t = mu_comp(E) + mu_phot(E) + mu_pair(E) + mu_rayl(E)
    r = np.random.random() * t
    if r < mu_comp(E):
        return 'compton'
    elif r < mu_comp(E) + mu_phot(E):
        return 'photoelectric'
    elif r < mu_comp(E) + mu_phot(E) + mu_pair(E):
        return 'pair'
    else:
        return 'rayleigh'

print("Testing source sampling...")
Nt = 10000
hits = 0
for _ in range(Nt):
    E, d, x, y = src_samp()
    # Ray from source through (x,y,zs) at z=zs
    # At z=zlo, the position is:
    # x_ep = x * (SSD + zl) / SSD = x * (100 + 4.9) / 100 = x * 1.049
    x_ep = x * (SSD + zlo) / SSD
    y_ep = y * (SSD + zlo) / SSD
    if np.sqrt(x_ep**2 + y_ep**2) <= Rc:
        hits += 1

print(f"Hit rate: {hits/Nt:.4f} (expected ~{np.pi*(Rc*SSD/(SSD+zlo))**2/(4*fh**2):.4f})")