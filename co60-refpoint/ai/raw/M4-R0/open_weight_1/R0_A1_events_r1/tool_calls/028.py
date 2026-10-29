# Continuing with the remaining functions

def comp_scat(E_in):
    a = E_in / mec2
    while True:
        ct = rng.uniform(-1.0, 1.0)
        E_out = E_in / (1.0 + a*(1.0-ct))
        if E_out <= 0 or E_out > E_in:
            continue
        r = E_out / E_in
        kn = r**2 * (r + 1.0/r - (1.0 - ct**2))
        if rng.random() * 2.0 < kn:
            te = E_in - E_out
            psq = te * (te + 2.0 * mec2)
            if psq <= 0:
                ce = 1.0
            else:
                ce = (E_in - E_out*ct) / np.sqrt(psq)
                ce = np.clip(ce, -1.0, 1.0)
            return E_out, ct, te, ce

def ray_scat(E_in):
    while True:
        ct = rng.uniform(-1.0, 1.0)
        if rng.random() < 0.5 * (1.0 + ct**2):
            return E_in, ct, 0.0, None

def pair_proc(E_in):
    exc = max(0.0, E_in - 1.022)
    te = rng.uniform(0.0, exc)
    tp = exc - te
    ce = rng.uniform(0.9, 1.0)
    return None, None, te, ce, tp, None

def perp_basis(v):
    v = v / np.linalg.norm(v)
    ex = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ex, v)) > 0.9:
        ex = np.array([0.0, 1.0, 0.0])
    ex = ex - np.dot(ex, v) * v
    ex = ex / np.linalg.norm(ex)
    ey = np.cross(v, ex)
    return ex, ey

def scatter_dir(di, ct):
    st = np.sqrt(1.0 - ct**2)
    phi = rng.random() * 2 * np.pi
    ex, ey = perp_basis(di)
    do = st * np.cos(phi) * ex + st * np.sin(phi) * ey + ct * di
    return do / np.linalg.norm(do)

print("Scattering functions defined")