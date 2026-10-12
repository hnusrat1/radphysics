import numpy as np
import time, math

me = 0.51099895
re = 2.8179403262e-13
Ne = 3.343e23

# Cross section tables
E_ph = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.15,0.20,0.30,0.50,0.80,1.0,1.5,2.0])
mu_ph= np.array([4.90,0.506,0.162,0.0712,0.0375,0.0223,0.00988,0.00547,0.00240,0.00128,0.000526,0.000151,5.0e-5,3.0e-5,1.0e-5,4.7e-6])
E_pa = np.array([1.022,1.25,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0,10.0])
mu_pa= np.array([0.0,3.1e-5,1.32e-4,4.1e-4,8.1e-4,1.39e-3,2.69e-3,4.32e-3,6.19e-3,1.05e-2,1.55e-2])
E_co = np.array([0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20,0.30,0.50,0.80,1.0,1.5,2.0])
mu_co= np.array([0.30,0.060,0.022,0.0098,0.0046,0.0034,0.0017,0.0011,0.0006,0.0003,0.00015,0.00012,8e-5,4.5e-5])
Egrid = np.logspace(np.log10(0.001), np.log10(10.0), 300)
def kn_sigma(E):
    mus = np.linspace(-1,1,3001)
    return np.pi*re**2*np.trapezoid(kn_f(mus,E),mus)
sig_grid = np.array([kn_sigma(E) for E in Egrid])

def kn_f(mu, E):
    a = E/me
    r = 1.0/(1.0+a*(1.0-mu))
    return r*r*(1.0/r + r - (1.0-mu*mu))

def mu_comp(E):
    s = float(np.interp(np.log(E), np.log(Egrid), sig_grid))
    return Ne * s
def mu_photo(E):
    return float(np.exp(np.interp(np.log(E), np.log(E_ph), np.log(mu_ph), left=np.log(mu_ph[0]), right=np.log(mu_ph[-1]))))
def mu_pair(E):
    if E <= 1.022: return 0.0
    return float(np.exp(np.interp(np.log(E), np.log(E_pa), np.log(mu_pa))))
def mu_coh(E):
    return float(np.exp(np.interp(np.log(E), np.log(E_co), np.log(mu_co))))
def mu_total(E):
    return mu_comp(E) + mu_photo(E) + mu_pair(E) + mu_coh(E)

# Sampling helpers
def sample_compton_single(E, rng):
    a = E/me
    while True:
        mu = rng.uniform(-1,1)
        f = kn_f(mu,E)
        if rng.uniform(0,2) < f:
            break
    Ep = E / (1.0 + a*(1.0-mu))
    te = E - Ep
    num = E - Ep*mu
    den = math.sqrt(E*E + Ep*Ep - 2.0*E*Ep*mu)
    cos_el = num/den if den > 1e-15 else 1.0
    return mu, Ep, te, cos_el

def direction_rotate(ux,uy,uz, mu_s, phi, rng):
    """Apply scattering with cos theta mu_s and azimuth phi. 
       Returns new direction. If mu_s close to 1, use phi from rng."""
    d = np.array([ux,uy,uz])
    d = d / np.linalg.norm(d)
    # orthonormal basis
    ref = np.array([0,0,1]) if abs(d[2]) < 0.9 else np.array([1,0,0])
    e1 = np.cross(d, ref)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    st = math.sqrt(max(0,1.0-mu_s*mu_s))
    newd = st*math.cos(phi)*e1 + st*math.sin(phi)*e2 + mu_s*d
    return newd[0], newd[1], newd[2]

# Geometry
def inside_cube(x,y,z):
    return abs(x)<15 and abs(y)<15 and 100<z<130

def in_scoring(x,y,z):
    return x*x+y*y<1.0 and 109.9<z<110.1

# Main transport loop per photon
def simulate_one(rng):
    """Simulate from source one primary photon, return list of events (interactions in scoring volume)"""
    events = []
    # initial state: sample entry point on z=100 within field
    x = rng.uniform(-5,5)
    y = rng.uniform(-5,5)
    z = 100.0
    # direction from source (0,0,0) to entry
    norm = math.sqrt(x*x+y*y+z*z)
    ux = x/norm; uy = y/norm; uz = z/norm
    E = 2.0
    pos = np.array([x,y,z], float)
    dir = np.array([ux,uy,uz], float)
    
    # queue of photons: list of (pos, dir, E)
    queue = [(pos.copy(), dir.copy(), E)]
    while queue:
        pos, dir, E = queue.pop(0)
        # track this photon until it leaves phantom or energy below cutoff
        while True:
            if E < 0.01:
                break
            if not inside_cube(pos[0], pos[1], pos[2]):
                break
            
            mu_tot = mu_total(E)
            if mu_tot <= 0: break
            # sample distance to next interaction
            s = -math.log(rng.uniform(1e-10,1)) / mu_tot
            # move
            pos[0] += s*dir[0]; pos[1] += s*dir[1]; pos[2] += s*dir[2]
            
            if not inside_cube(pos[0], pos[1], pos[2]):
                break  # photon leaves phantom
            
            # interaction at pos
            if in_scoring(pos[0], pos[1], pos[2]):
                # record this event
                mu_c = mu_comp(E); mu_ph = mu_photo(E); mu_pa = mu_pair(E); mu_co = mu_coh(E)
                tot = mu_c+mu_ph+mu_pa+mu_co
                u = rng.uniform(0,tot)
                if u <= mu_c:
                    # Compton
                    mu_s, Ep, te, cos_el = sample_compton_single(E, rng)
                    events.append({'process':'compton', 'e_in_MeV':E, 'e_out_MeV':Ep,
                                   'cos_scatter':mu_s, 't_e_MeV':te, 'cos_electron':cos_el,
                                   't_positron_MeV':None})
                    # Update state: new energy, direction
                    phi = rng.uniform(0,2*math.pi)
                    new_dir = direction_rotate(dir[0], dir[1], dir[2], mu_s, phi, rng)
                    dir = np.array(new_dir)
                    E = Ep
                    continue  # continue tracking scattered photon
                    
                elif u <= mu_c+mu_ph:
                    # photoelectric
                    te = E  # approx
                    cos_el = rng.uniform(-1,1)
                    events.append({'process':'photoelectric', 'e_in_MeV':E, 'e_out_MeV':None,
                                   'cos_scatter':None, 't_e_MeV':te, 'cos_electron':cos_el,
                                   't_positron_MeV':None})
                    break  # photon absorbed
                    
                elif u <= mu_c+mu_ph+mu_pa:
                    # pair production
                    t_avail = E - 1.022
                    share = rng.uniform(0,1)
                    t_pos = share*t_avail
                    t_e = t_avail - t_pos
                    # electron angle forward peaked
                    E_total_e = t_e + me
                    theta0_e = me / E_total_e if E_total_e>0 else 0.5
                    # sample theta from exponential with scale theta0_e
                    theta_e = theta0_e * (-math.log(rng.uniform(1e-10,1)))
                    cos_el = math.cos(theta_e)
                    events.append({'process':'pair', 'e_in_MeV':E, 'e_out_MeV':None,
                                   'cos_scatter':None, 't_e_MeV':t_e, 'cos_electron':cos_el,
                                   't_positron_MeV':t_pos})
                    # Emit two annihilation photons
                    for _ in range(2):
                        u1 = rng.uniform(-1,1)
                        u2 = rng.uniform(0,2*math.pi)
                        nz = u1; nx = math.sqrt(1-u1*u1)*math.cos(u2); ny = math.sqrt(1-u1*u1)*math.sin(u2)
                        ann_pos = pos.copy()
                        ann_dir = np.array([nx,ny,nz])
                        queue.append((ann_pos, ann_dir, 0.511))
                    break  # pair absorbed photon
                    
                else:
                    # Rayleigh
                    theta0 = min(0.5, 0.0007/E)  # characteristic angle rad
                    theta = theta0 * (-math.log(rng.uniform(1e-10,1)))
                    mu_s = math.cos(theta)
                    cos_el = None
                    events.append({'process':'rayleigh', 'e_in_MeV':E, 'e_out_MeV':E,
                                   'cos_scatter':mu_s, 't_e_MeV':0.0, 'cos_electron':None,
                                   't_positron_MeV':None})
                    # Update direction
                    phi = rng.uniform(0,2*math.pi)
                    new_dir = direction_rotate(dir[0], dir[1], dir[2], mu_s, phi, rng)
                    dir = np.array(new_dir)
                    # energy unchanged
                    continue
            else:
                # interaction NOT in scoring - update state and continue
                mu_c = mu_comp(E); mu_ph = mu_photo(E); mu_pa = mu_pair(E); mu_co = mu_coh(E)
                tot = mu_c+mu_ph+mu_pa+mu_co
                u = rng.uniform(0,tot)
                if u <= mu_c:
                    mu_s, Ep, te, cos_el = sample_compton_single(E, rng)
                    phi = rng.uniform(0,2*math.pi)
                    new_dir = direction_rotate(dir[0], dir[1], dir[2], mu_s, phi, rng)
                    dir = np.array(new_dir)
                    E = Ep
                    continue
                elif u <= mu_c+mu_ph:
                    break  # photo absorbed
                elif u <= mu_c+mu_ph+mu_pa:
                    t_avail = E - 1.022
                    t_pos = rng.uniform(0,1)*t_avail
                    t_e = t_avail - t_pos
                    for _ in range(2):
                        u1 = rng.uniform(-1,1); u2 = rng.uniform(0,2*math.pi)
                        nz=u1; nx=math.sqrt(1-u1*u1)*math.cos(u2); ny=math.sqrt(1-u1*u1)*math.sin(u2)
                        queue.append((pos.copy(), np.array([nx,ny,nz]), 0.511))
                    break
                else:
                    # Rayleigh
                    theta0 = min(0.5, 0.0007/E)
                    theta = theta0 * (-math.log(rng.uniform(1e-10,1)))
                    mu_s = math.cos(theta)
                    phi = rng.uniform(0,2*math.pi)
                    new_dir = direction_rotate(dir[0], dir[1], dir[2], mu_s, phi, rng)
                    dir = np.array(new_dir)
                    continue
        # end while for this photon
    return events

# Test timing with 200 primaries
rng = np.random.default_rng(123)
t0 = time.time()
all_events = []
for i in range(200):
    ev = simulate_one(rng)
    all_events.extend(ev)
t1 = time.time()
print(f"200 primaries: {len(all_events)} events in {t1-t0:.2f}s")
if all_events:
    print("Sample event:", all_events[0])