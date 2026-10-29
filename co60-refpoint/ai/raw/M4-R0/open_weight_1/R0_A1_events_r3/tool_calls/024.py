import numpy as np

# Constants
m_e = 0.510998950
r_e = 2.8179403262e-13
NA = 6.02214076e23
e_per_cm3 = 10.0 * NA / 18.01528

# Cross section functions
def mu_compton(E):
    a = E / m_e
    t1 = (1+a)/a**2 * (2*(1+a)/(1+2*a) - np.log(1+2*a)/a)
    t2 = np.log(1+2*a)/(2*a)
    t3 = (1+3*a)/(1+2*a)**2
    return 2*np.pi*r_e**2 * (t1 + t2 - t3) * e_per_cm3

def mu_photo(E):
    xs = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.1732, 1.25, 1.3325, 1.5, 2.0])
    ys = np.array([4.91, 1.67, 0.755, 0.216, 0.0881, 0.0442, 0.0255, 0.0113, 0.00642, 0.00254, 0.00152, 0.00154, 0.00228, 0.00406, 0.00555, 0.00450, 0.00377, 0.00300, 0.00258, 0.00230, 0.00179, 0.00100])
    return np.interp(E, xs, ys) * 1.0

def mu_pair(E):
    if E < 2*m_e: return 0.0
    xs = np.array([1.022, 1.1, 1.1732, 1.25, 1.3325, 1.5, 2.0, 3.0, 5.0])
    ys = np.array([0.0, 0.0002, 0.00035, 0.00071, 0.00100, 0.00146, 0.00383, 0.0085, 0.0175])
    return np.interp(E, xs, ys) * 1.0

def mu_rayleigh(E):
    xs = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.1732, 1.25, 1.3325, 1.5, 2.0])
    ys = np.array([0.037, 0.034, 0.032, 0.028, 0.025, 0.022, 0.019, 0.015, 0.012, 0.0085, 0.0065, 0.0050, 0.0040, 0.0055, 0.0045, 0.0035, 0.00303, 0.00250, 0.00242, 0.00210, 0.00186, 0.00114])
    return np.interp(E, xs, ys) * 1.0

def mu_total(E):
    return mu_compton(E) + mu_photo(E) + mu_pair(E) + mu_rayleigh(E)

def sample_type(E):
    mc = mu_compton(E); mp = mu_photo(E); mpp = mu_pair(E); mr = mu_rayleigh(E)
    total = mc + mp + mpp + mr
    r = np.random.random() * total
    if r < mc: return 'compton'
    r -= mc
    if r < mp: return 'photoelectric'
    r -= mp
    if r < mpp: return 'pair'
    return 'rayleigh'

def sample_compton(E_in):
    a = E_in / m_e
    while True:
        cos = 2.0*np.random.random() - 1.0
        E_out = E_in / (1.0 + a * (1.0 - cos))
        kn = (E_out/E_in)**2 * (E_out/E_in + E_in/E_out - 1.0 + cos**2)
        if np.random.random() <= kn:
            return cos, E_out, E_in - E_out

def sample_rayleigh(E_in):
    while True:
        theta = np.random.exponential(0.002)
        if theta > 0.2: continue
        cos = np.cos(theta)
        ff2 = np.exp(-(theta/0.001)**2)
        w = (1.0 + cos**2) / 2.0 * ff2
        if np.random.random() <= w:
            return cos, E_in

def in_scoring(pos):
    x, y, z = pos
    return (4.9 <= z <= 5.1) and (x*x + y*y <= 1.0)

def in_phantom(pos):
    x, y, z = pos
    return (0.0 <= z <= 30.0) and (-15.0 <= x <= 15.0) and (-15.0 <= y <= 15.0)

def dist_to_boundary(pos, dir):
    d = np.inf
    for z0 in (0.0, 30.0):
        if abs(dir[2]) > 1e-12:
            dz = (z0 - pos[2]) / dir[2]
            if dz > 1e-12 and dz < d: d = dz
    for x0 in (-15.0, 15.0):
        if abs(dir[0]) > 1e-12:
            dx = (x0 - pos[0]) / dir[0]
            if dx > 1e-12 and dx < d: d = dx
    for y0 in (-15.0, 15.0):
        if abs(dir[1]) > 1e-12:
            dy = (y0 - pos[1]) / dir[1]
            if dy > 1e-12 and dy < d: d = dy
    return d

def sample_source():
    E = 1.1732 if np.random.random() < 0.5 else 1.3325
    x = (np.random.random() - 0.5) * 10.0
    y = (np.random.random() - 0.5) * 10.0
    d = np.array([x, y, 100.0])
    d = d / np.linalg.norm(d)
    pos = np.array([x, y, 0.0])
    return E, pos, d

print("All functions defined.")
print(f"mu_total(1.1732,)={mu_total(1.1732):.6f}")
print(f"mu_total(1.3325,)={mu_total(1.3325):.6f}")