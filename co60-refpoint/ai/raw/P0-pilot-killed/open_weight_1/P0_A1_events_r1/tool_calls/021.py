# Let's generate the events and get the JSON string length.
# We'll use the same code as before but with smaller number of primaries to quickly get 200 events.
# We'll reuse the function definitions from earlier (but they are not in this session). Let's redefine quickly.
import numpy as np
import math
from scipy.interpolate import interp1d
import json

me = 0.511
r_e = 2.817940e-13

energies = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
mu_coh = np.array([0.03094, 0.01858, 0.01234, 0.006699, 0.003458, 0.002573, 0.001571, 0.001105, 0.00063, 0.00039, 0.00027, 0.00019, 0.00012, 0.00008])
mu_inc = np.array([0.1237, 0.1153, 0.1042, 0.08908, 0.07439, 0.06756, 0.05565, 0.04748, 0.03454, 0.02668, 0.02087, 0.01681, 0.01162, 0.00883])
mu_ph = np.array([0.00627, 0.00260, 0.00125, 0.00047, 0.00014, 0.000089, 0.000029, 0.000013, 0.000004, 0.0000017, 8e-7, 0.0, 0.0, 0.0])
mu_pair = np.array([0.0, 0.0, 0.0, 0.0, 0.00015, 0.00046, 0.00150, 0.00177, 0.00345, 0.00517, 0.00696, 0.00835, 0.0106, 0.0118])

def interp_log(x, y):
    mask = y > 0
    x_ok = x[mask]
    y_ok = y[mask]
    logx = np.log(x_ok)
    logy = np.log(y_ok)
    f = interp1d(logx, logy, kind='linear', fill_value='extrapolate')
    def fun(xi):
        if xi <= x_ok[0]: return y_ok[0]
        if xi >= x_ok[-1]: return y_ok[-1]
        return np.exp(f(np.log(xi)))
    return fun

mu_coh_func = interp_log(energies, mu_coh)
mu_inc_func = interp_log(energies, mu_inc)
mu_ph_func = interp_log(energies, mu_ph)
mu_pair_func = interp_log(energies, mu_pair)

def mu_total(E):
    return mu_coh_func(E) + mu_inc_func(E) + mu_ph_func(E) + mu_pair_func(E)

def sample_compton(E):
    alpha = E / me
    f_max = 2 * math.pi * r_e**2
    while True:
        u = np.random.uniform(-1, 1)
        k = alpha
        kp = k / (1 + k * (1 - u))
        sin2 = 1 - u*u
        f = 2*math.pi * (r_e**2/2) * (kp/k)**2 * (kp/k + k/kp - sin2)
        if f < 0: continue
        y = np.random.uniform(0, f_max)
        if y <= f:
            cos_theta = u
            break
    E_prime = E / (1 + alpha * (1 - cos_theta))
    T = E - E_prime
    p = E
    p_prime = E_prime
    if T > 0:
        p_e = math.sqrt(T*T + 2*me*T)
        cos_e = (p - p_prime * cos_theta) / p_e
        if cos_e > 1: cos_e = 1.0
        if cos_e < -1: cos_e = -1.0
    else:
        cos_e = 1.0
    return cos_theta, E_prime, T, cos_e

def sample_rayleigh(E):
    sigma = 0.01
    angle = abs(np.random.standard_normal()) * sigma
    cos_theta = math.cos(angle)
    return cos_theta, E, 0.0

def sample_photoelectric(E):
    return E, 1.0

def sample_pair(E):
    T_avail = E - 2*me
    T_e = T_avail / 2.0
    T_pos = T_avail - T_e
    return T_e, 1.0, T_pos

def in_phantom(pos):
    x,y,z = pos
    return -15 <= x <= 15 and -15 <= y <= 15 and 0 <= z <= 30

scoring_zmin = 9.9
scoring_zmax = 10.1
scoring_radius = 1.0

def in_scoring(pos):
    x,y,z = pos
    if z < scoring_zmin or z > scoring_zmax: return False
    return x*x + y*y <= scoring_radius*scoring_radius

def sample_primary():
    x = np.random.uniform(-5, 5)
    y = np.random.uniform(-5, 5)
    source = np.array([0.0, 0.0, -100.0])
    target = np.array([x, y, 0.0])
    dir_vec = target - source
    dir = dir_vec / np.linalg.norm(dir_vec)
    entry = target
    return 2.0, entry, dir

def track_photon(E, pos, dir, stack, events, max_events):
    while True:
        if not in_phantom(pos): return
        mu = mu_total(E)
        if mu <= 0: return
        s = -math.log(np.random.random()) / mu
        new_pos = pos + s * dir
        if not in_phantom(new_pos): return
        pos = new_pos
        p_coh = mu_coh_func(E)
        p_inc = mu_inc_func(E)
        p_ph = mu_ph_func(E)
        p_pair = mu_pair_func(E)
        total = p_coh + p_inc + p_ph + p_pair
        r = np.random.random() * total
        
        if r < p_coh:
            process = 'rayleigh'
            cos_scat, e_out, T_e = sample_rayleigh(E)
            cos_el = None; T_pos = None
        elif r < p_coh + p_inc:
            process = 'compton'
            cos_scat, e_out, T_e, cos_el = sample_compton(E)
            T_pos = None
        elif r < p_coh + p_inc + p_ph:
            process = 'photoelectric'
            T_e, cos_el = sample_photoelectric(E)
            e_out = None; cos_scat = None; T_pos = None
        else:
            process = 'pair'
            T_e, cos_el, T_pos = sample_pair(E)
            e_out = None; cos_scat = None
        
        if in_scoring(pos):
            events.append({
                'process': process,
                'e_in_MeV': E,
                'e_out_MeV': e_out,
                'cos_scatter': cos_scat,
                't_e_MeV': T_e,
                'cos_electron': cos_el,
                't_positron_MeV': T_pos
            })
            if len(events) >= max_events: return True
        
        if e_out is not None and process in ('compton','rayleigh'):
            if cos_scat is not None and cos_scat < 1.0:
                angle = math.acos(cos_scat)
                phi = np.random.random() * 2*math.pi
                up = np.array([0,0,1])
                if abs(np.dot(dir, up)) > 0.99: up = np.array([1,0,0])
                perp = np.cross(dir, up)
                perp = perp / np.linalg.norm(perp)
                new_dir = dir * math.cos(angle) + np.cross(perp, dir) * math.sin(angle) + perp * np.dot(perp, dir) * (1 - math.cos(angle))
            else:
                new_dir = dir
            stack.append((e_out, pos, new_dir))
        return

def simulate(num_primaries, max_events=200):
    events = []
    stack = []
    for i in range(num_primaries):
        if len(events) >= max_events: break
        E, pos, dir = sample_primary()
        track_photon(E, pos, dir, stack, events, max_events)
        while stack and len(events) < max_events:
            E, pos, dir = stack.pop()
            track_photon(E, pos, dir, stack, events, max_events)
    return events

np.random.seed(12345)
events = simulate(800000, 200)
output = {"method": "Monte Carlo simulation ...", "events": events}
compact = json.dumps(output, separators=(',', ':'))
print(len(compact))