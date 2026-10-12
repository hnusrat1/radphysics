import numpy as np
import math
from scipy.interpolate import interp1d

# Constants
rho = 1.0
me = 0.511
r_e = 2.817940e-13

# Cross sections for water
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
        if xi <= x_ok[0]:
            return y_ok[0]
        if xi >= x_ok[-1]:
            return y_ok[-1]
        return np.exp(f(np.log(xi)))
    return fun

mu_coh_func = interp_log(energies, mu_coh)
mu_inc_func = interp_log(energies, mu_inc)
mu_ph_func = interp_log(energies, mu_ph)
mu_pair_func = interp_log(energies, mu_pair)

def mu_total(E):
    return mu_coh_func(E) + mu_inc_func(E) + mu_ph_func(E) + mu_pair_func(E)

# Compton sampling
def sample_compton(E):
    alpha = E / me
    f_max = 2 * math.pi * r_e**2
    while True:
        u = np.random.uniform(-1, 1)
        k = alpha
        kp = k / (1 + k * (1 - u))
        sin2 = 1 - u*u
        f = 2*math.pi * (r_e**2/2) * (kp/k)**2 * (kp/k + k/kp - sin2)
        if f < 0:
            continue
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

# Rayleigh sampling (very small angle)
def sample_rayleigh(E):
    sigma_angle = 0.01
    angle = abs(np.random.standard_normal()) * sigma_angle
    cos_theta = math.cos(angle)
    E_prime = E
    T = 0.0
    return cos_theta, E_prime, T

# Photoelectric sampling
def sample_photoelectric(E):
    T = E  # ignore binding
    cos_e = 1.0
    return T, cos_e

# Pair production sampling
def sample_pair(E):
    T_avail = E - 2*me
    T_e = T_avail / 2.0
    T_pos = T_avail - T_e
    cos_e = 1.0
    return T_e, cos_e, T_pos

# Geometry
def in_phantom(pos):
    x,y,z = pos
    if x < -15 or x > 15 or y < -15 or y > 15 or z < 0 or z > 30:
        return False
    return True

scoring_center = np.array([0.0, 0.0, 10.0])
scoring_radius = 1.0
scoring_thickness = 0.2
scoring_zmin = 9.9
scoring_zmax = 10.1

def in_scoring(pos):
    x,y,z = pos
    if z < scoring_zmin or z > scoring_zmax:
        return False
    if x*x + y*y > scoring_radius*scoring_radius:
        return False
    return True

# Sample primary photon direction to pass through the scoring volume
def sample_primary_directed():
    # sample a point in the scoring cylinder at z=10 (mid)
    # actually we want the line to go through the volume. We'll sample a point on the midplane of the scoring volume.
    x = np.random.uniform(-scoring_radius, scoring_radius)
    y = np.random.uniform(-scoring_radius, scoring_radius)
    while x*x + y*y > scoring_radius*scoring_radius:
        x = np.random.uniform(-scoring_radius, scoring_radius)
        y = np.random.uniform(-scoring_radius, scoring_radius)
    # point at z=10
    target = np.array([x, y, 10.0])
    source = np.array([0.0, 0.0, -100.0])
    dir_vec = target - source
    dir = dir_vec / np.linalg.norm(dir_vec)
    # entry point at z=0
    # param t such that z = -100 + t * dir_z = 0 => t = 100 / dir_z
    t = 100.0 / dir[2]
    entry = source + t * dir
    return source, dir, entry

# Main simulation using directed primaries and collecting 200 events
def simulate_directed(target_events=200, max_primaries=100000):
    events = []
    primary_count = 0
    # We'll also track scattered photons from interactions
    stack = []
    while len(events) < target_events and primary_count < max_primaries:
        # generate a primary photon directed through scoring volume
        source, dir, entry = sample_primary_directed()
        stack.append((2.0, entry, dir, True))
        primary_count += 1
        while stack and len(events) < target_events:
            E, pos, dir, is_prim = stack.pop()
            if not in_phantom(pos):
                continue
            mu = mu_total(E)
            # track
            while True:
                if mu <= 0:
                    break
                s = -math.log(np.random.random()) / mu
                new_pos = pos + s * dir
                if not in_phantom(new_pos):
                    # find intersection with boundary? skip
                    break
                pos = new_pos
                # interaction occurs here
                # choose process
                p_coh = mu_coh_func(E)
                p_inc = mu_inc_func(E)
                p_ph = mu_ph_func(E)
                p_pair = mu_pair_func(E)
                total = p_coh + p_inc + p_ph + p_pair
                r = np.random.random() * total
                # default values
                process = None
                e_out = None
                cos_scat = None
                T_e = 0.0
                cos_el = None
                T_pos = None
                secondary_photon = None
                if r < p_coh:
                    # Rayleigh
                    process = 'rayleigh'
                    cos_scat, e_out, T_e = sample_rayleigh(E)
                    if e_out is not None:
                        secondary_photon = (e_out, pos, None)  # direction update later
                elif r < p_coh + p_inc:
                    # Compton
                    process = 'compton'
                    cos_scat, e_out, T_e, cos_el = sample_compton(E)
                    if e_out is not None:
                        secondary_photon = (e_out, pos, None)
                elif r < p_coh + p_inc + p_ph:
                    # Photoelectric
                    process = 'photoelectric'
                    T_e, cos_el = sample_photoelectric(E)
                    # no secondary photon
                else:
                    # Pair production
                    process = 'pair'
                    T_e, cos_el, T_pos = sample_pair(E)
                    # no secondary photon (annihilation not tracked)
                if process is None:
                    break
                # Record if in scoring volume
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
                # push secondary photon if any
                if secondary_photon is not None:
                    e_sec, pos_sec, _ = secondary_photon
                    # compute new direction based on scattering angle
                    if cos_scat is not None and cos_scat < 1.0:
                        angle = math.acos(cos_scat)
                        phi = np.random.random() * 2*math.pi
                        up = np.array([0,0,1])
                        if abs(np.dot(dir, up)) > 0.99:
                            up = np.array([1,0,0])
                        perp = np.cross(dir, up)
                        perp = perp / np.linalg.norm(perp)
                        new_dir = dir * math.cos(angle) + np.cross(perp, dir) * math.sin(angle) + perp * np.dot(perp, dir) * (1 - math.cos(angle))
                    else:
                        new_dir = dir
                    stack.append((e_sec, pos_sec, new_dir, False))
                # after interaction, current photon is done
                break
    return events, primary_count

np.random.seed(42)
events, prim = simulate_directed(200, 50000)
print("Primary attempts:", prim)
print("Events:", len(events))
for e in events[:5]:
    print(e)