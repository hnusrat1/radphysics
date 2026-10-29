with open('/tmp/full_sim.py', 'a') as f:
    f.write('''
    else:
        w = np.cross(u, np.array([1.0, 0.0, 0.0]))
    w = w / np.linalg.norm(w)
    vv = np.cross(w, u)
    return u*ct + w*np.sin(thv)*np.cos(ph) + vv*np.sin(thv)*np.sin(ph)

def src_samp():
    E = 1.1732 if np.random.random() < 0.5 else 1.3325
    x = np.random.uniform(-fh, fh)
    y = np.random.uniform(-fh, fh)
    d = np.array([x, y, SSD])
    d = d / np.linalg.norm(d)
    return E, d, x, y

def sample_proc(E):
    t = mu_c(E) + mu_ph(E) + mu_pp(E) + mu_rl(E)
    r = np.random.random() * t
    if r < mu_c(E):
        return "compton"
    elif r < mu_c(E) + mu_ph(E):
        return "photoelectric"
    elif r < mu_c(E) + mu_ph(E) + mu_pp(E):
        return "pair"
    else:
        return "rayleigh"

def intersect_cyl(ori, direc, R, zL, zH):
    d = direc / np.linalg.norm(direc)
    if d[2] > 0:
        tzL = (zL - ori[2]) / d[2]
        tzH = (zH - ori[2]) / d[2]
    elif d[2] < 0:
        tzL = (zH - ori[2]) / d[2]
        tzH = (zL - ori[2]) / d[2]
    else:
        if ori[2] < zL or ori[2] > zH:
            return None
        tzL = -1e30
        tzH = 1e30
    a = d[0]**2 + d[1]**2
    if a < 1e-15:
        if ori[0]**2 + ori[1]**2 <= R**2:
            return (tzL, tzH) if tzL < tzH else (tzH, tzL)
        return None
    b = 2.0 * (ori[0]*d[0] + ori[1]*d[1])
    c = ori[0]**2 + ori[1]**2 - R**2
    disc = b*b - 4.0*a*c
    if disc < 0:
        return None
    sq = np.sqrt(disc)
    t1 = (-b - sq) / (2.0*a)
    t2 = (-b + sq) / (2.0*a)
    if t1 > t2:
        t1, t2 = t2, t1
    te = max(tzL, t1)
    tx = min(tzH, t2)
    if te >= tx or tx < 0:
        return None
    return (te, tx)

# ===== COMPUTE RATES =====
NMC = 500000

# Primary rate
prim_tot = 0.0
prim_by = {"compton":0.0, "photoelectric":0.0, "pair":0.0, "rayleigh":0.0}

for _ in range(NMC):
    E, d, x, y = src_samp()
    dz = d[2]
    if dz <= 0:
        continue
    x_ep = x * (SSD + zlo) / SSD
    y_ep = y * (SSD + zlo) / SSD
    if np.sqrt(x_ep**2 + y_ep**2) > Rc:
        continue
    path = 0.2 / dz
    muE = mu_t(E)
    t_sc = (zlo - zs) / dz
    att = np.exp(-muE * t_sc)
    p_int = 1.0 - np.exp(-muE * path)
    w = att * p_int
    prim_tot += w
    mt = mu_c(E) + mu_ph(E) + mu_pp(E) + mu_rl(E)
    prim_by["compton"] += w * mu_c(E) / mt
    prim_by["photoelectric"] += w * mu_ph(E) / mt
    prim_by["pair"] += w * mu_pp(E) / mt
    prim_by["rayleigh"] += w * mu_rl(E) / mt

prim_tot /= NMC
for k in prim_by:
    prim_by[k] /= NMC

print(f"Primary: {prim_tot:.8f}", file=sys.stderr)
for k, v in prim_by.items():
    print(f"  {k}: {v:.10f}", file=sys.stderr)

# Scatter rate (forced detection)
scat_tot = 0.0
scat_by = {"compton":0.0, "photoelectric":0.0, "pair":0.0, "rayleigh":0.0}

for _ in range(NMC):
    E, d, x, y = src_samp()
    dz = d[2]
    if dz <= 0:
        continue
    muE = mu_t(E)
    t_max = 30.0 / dz
    C = 1.0 - np.exp(-muE * t_max)
    if C <= 0:
        continue
    t_scat = -np.log(1.0 - np.random.random() * C) / muE
    sp = np.array([x, y, zs]) + t_scat * d
    if abs(sp[0]) > 15 or abs(sp[1]) > 15:
        continue
    pdf_scat = muE * np.exp(-muE * t_scat) / C

    # Direction to scoring volume center
    d2v = np.array([0.0, 0.0, zc]) - sp
    dist = np.linalg.norm(d2v)
    if dist < 0.01:
        continue
    d2v = d2v / dist
    costh = np.dot(d, d2v)
    costh = np.clip(costh, -1.0, 1.0)
    dcs = KN(E, costh)
    E_out = c_E(E, costh)

    # Solid angle of scoring volume
    cosv = abs(d2v[2])
    proj = np.pi * Rc**2 * max(cosv, 0.01)
    solid = proj / (dist**2)

    # Check ray hits scoring volume
    it = intersect_cyl(sp, d2v, Rc, zlo, zhi)
    if it is None:
        continue
    te, tx = it
    if te < 0:
        te = 0
    pv = tx - te
    if pv <= 0:
        continue

    mu_out = mu_t(E_out)
    att_v = np.exp(-mu_out * te)
    p_int_v = 1.0 - np.exp(-mu_out * pv)

    w = ne * dcs * solid * att_v * p_int_v / pdf_scat
    scat_tot += w
    mt = mu_c(E_out) + mu_ph(E_out) + mu_pp(E_out) + mu_rl(E_out)
    scat_by["compton"] += w * mu_c(E_out) / mt
    scat_by["photoelectric"] += w * mu_ph(E_out) / mt
    scat_by["pair"] += w * mu_pp(E_out) / mt
    scat_by["rayleigh"] += w * mu_rl(E_out) / mt

scat_tot /= NMC
for k in scat_by:
    scat_by[k] /= NMC

print(f"Scatter: {scat_tot:.8f}", file=sys.stderr)
for k, v in scat_by.items():
    print(f"  {k}: {v:.10f}", file=sys.stderr)

tot = prim_tot + scat_tot
print(f"Total: {tot:.8f}, scatter frac: {scat_tot/tot:.4f}", file=sys.stderr)

# ===== GENERATE 200 EVENTS =====
n_prim = int(round(200 * prim_tot / tot))
n_scat = 200 - n_prim
print(f"Generating {n_prim} primary + {n_scat} scatter", file=sys.stderr)
events = []

# Primary events
gen = 0; atts = 0; max_a = 5000000
while gen < n_prim and atts < max_a:
    atts += 1
    E, d, x, y = src_samp()
    dz = d[2]
    if dz <= 0: continue
    x_ep = x * (SSD + zlo) / SSD
    y_ep = y * (SSD + zlo) / SSD
    if np.sqrt(x_ep**2 + y_ep**2) > Rc: continue
    path = 0.2 / dz
    muE = mu_t(E)
    t_sc = (zlo - zs) / dz
    att = np.exp(-muE * t_sc)
    p_int = 1.0 - np.exp(-muE * path)
    max_p = np.exp(-0.064 * 5.0) * (1.0 - np.exp(-0.064 * 0.2))
    if np.random.random() * max_p * 1.5 > att * p_int: continue
    
    proc = sample_proc(E)
''')
print("File written so far")