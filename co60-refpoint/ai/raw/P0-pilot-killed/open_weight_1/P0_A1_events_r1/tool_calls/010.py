import numpy as np
import math
from scipy.interpolate import interp1d

# Constants
rho = 1.0  # g/cm3
me = 0.511  # MeV
r_e = 2.817940e-13  # cm
# Cross sections for water (NIST XCOM)
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
    # rejection maximum
    f_max = 2 * math.pi * r_e**2  # actually we need max of f(u)
    # But the actual maximum is at u=1, f(1)=2π * (r_e^2/2)*2 = 2π r_e^2
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
    # electron direction
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

# Rayleigh sampling (very forward)
def sample_rayleigh(E):
    # For simplicity, set cosθ = 1 (no deflection). But we need a small angle.
    # Use a Gaussian distribution with sigma = 0.01 rad (approx 0.57 deg)
    sigma_angle = 0.01
    angle = np.random.standard_normal() * sigma_angle
    angle = abs(angle)  # small positive
    cos_theta = math.cos(angle)
    # no energy loss
    E_prime = E
    T = 0.0
    cos_e = None
    return cos_theta, E_prime, T, cos_e

# Photoelectric sampling
def sample_photoelectric(E):
    # Electron kinetic energy = E - binding, ignore binding
    T = E
    # Angular distribution: forward peaked. Simplified: cos_electron = 1
    cos_e = 1.0
    # No outgoing photon
    return None, None, T, cos_e

# Pair production sampling
def sample_pair(E):
    # Available kinetic energy
    T_avail = E - 2*me
    # Assume equal sharing
    T_e = T_avail / 2.0
    T_pos = T_avail - T_e
    # Electron forward
    cos_e = 1.0
    # No outgoing photon
    return None, None, T_e, cos_e, T_pos

# Geometry
phantom_xmin, phantom_xmax = -15.0, 15.0
phantom_ymin, phantom_ymax = -15.0, 15.0
phantom_zmin, phantom_zmax = 0.0, 30.0

# Scoring volume
scoring_center = np.array([0.0, 0.0, 10.0])
scoring_radius = 1.0  # cm
scoring_thickness = 0.2  # cm
scoring_zmin = scoring_center[2] - scoring_thickness/2.0
scoring_zmax = scoring_center[2] + scoring_thickness/2.0

def in_scoring_volume(pos):
    x,y,z = pos
    if z < scoring_zmin or z > scoring_zmax:
        return False
    if x*x + y*y > scoring_radius*scoring_radius:
        return False
    return True

def in_phantom(pos):
    x,y,z = pos
    if x < phantom_xmin or x > phantom_xmax:
        return False
    if y < phantom_ymin or y > phantom_ymax:
        return False
    if z < phantom_zmin or z > phantom_zmax:
        return False
    return True

# Source sampling
def sample_primary_photon():
    # point source at (0,0,-100)
    source = np.array([0.0, 0.0, -100.0])
    # field at z=0: square 10x10 centered at 0,0
    x_surf = np.random.uniform(-5.0, 5.0)
    y_surf = np.random.uniform(-5.0, 5.0)
    # direction from source to surface point
    dir_vec = np.array([x_surf, y_surf, 0.0]) - source
    dir_norm = np.linalg.norm(dir_vec)
    dir = dir_vec / dir_norm
    # entry point at z=0 (surface)
    entry = np.array([x_surf, y_surf, 0.0])
    return source, dir, entry

# Main simulation
def simulate(num_primary):
    events = []
    # stack: each item is (E, pos, dir, primary_flag)
    stack = []
    for i in range(num_primary):
        if len(events) >= 200:
            break
        source, dir, entry = sample_primary_photon()
        # start with primary photon
        stack.append((2.0, entry, dir, True))
        while stack and len(events) < 200:
            E, pos, dir, is_prim = stack.pop()
            # track from pos in direction dir
            # check if already outside phantom? pos should be inside when created
            # compute path lengths
            # we need to sample interaction distance
            # but first check if the photon can exit without interacting
            mu = mu_total(E)
            while True:
                # sample distance to next interaction
                if mu > 0:
                    s = -math.log(np.random.random()) / mu
                else:
                    break  # no interaction
                new_pos = pos + s * dir
                # if new position outside phantom, the photon exits
                if not in_phantom(new_pos):
                    # moves to exit
                    # find intersection point with boundary? Not needed.
                    break
                # interaction occurred at new_pos
                pos = new_pos
                # check if scoring volume
                if in_scoring_volume(pos):
                    # record interaction
                    # choose process
                    p_coh = mu_coh_func(E)
                    p_inc = mu_inc_func(E)
                    p_ph = mu_ph_func(E)
                    p_pair = mu_pair_func(E)
                    total = p_coh + p_inc + p_ph + p_pair
                    r = np.random.random() * total
                    if r < p_coh:
                        # Rayleigh
                        cos_scat, e_out, T_e, cos_el = sample_rayleigh(E)
                        # t_positron = null
                        events.append({
                            'process': 'rayleigh',
                            'e_in_MeV': E,
                            'e_out_MeV': e_out,
                            'cos_scatter': cos_scat,
                            't_e_MeV': T_e,
                            'cos_electron': None,
                            't_positron_MeV': None
                        })
                        if e_out is not None and E > 0.01:
                            # scattered photon continues with new direction
                            # update direction (for Rayleigh, direction change is small)
                            # For simplicity, we assume direction unchanged (cosθ=1)
                            # Actually our Rayleigh sampling gave cos_theta maybe not 1, but we need to update dir.
                            # We'll compute new dir by rotating old dir by the scattering angle.
                            if cos_scat < 1.0:
                                # rotate
                                angle = math.acos(cos_scat)
                                # random azimuth
                                phi = np.random.random() * 2*math.pi
                                # we need to rotate around an axis perpendicular to dir
                                # pick a perpendicular axis
                                # find a vector not parallel to dir
                                up = np.array([0,0,1])
                                if abs(np.dot(dir, up)) > 0.99:
                                    up = np.array([1,0,0])
                                perp = np.cross(dir, up)
                                perp = perp / np.linalg.norm(perp)
                                # rotation matrix
                                # rotate dir by angle around perp
                                # Rodrigues formula
                                dir_new = dir * math.cos(angle) + np.cross(perp, dir) * math.sin(angle) + perp * np.dot(perp, dir) * (1 - math.cos(angle))
                            else:
                                dir_new = dir
                            # push scattered photon
                            stack.append((e_out, pos, dir_new, False))
                    elif r < p_coh + p_inc:
                        # Compton
                        cos_scat, e_out, T_e, cos_el = sample_compton(E)
                        events.append({
                            'process': 'compton',
                            'e_in_MeV': E,
                            'e_out_MeV': e_out,
                            'cos_scatter': cos_scat,
                            't_e_MeV': T_e,
                            'cos_electron': cos_el,
                            't_positron_MeV': None
                        })
                        if e_out is not None and E > 0.01:
                            # scatter photon direction
                            if cos_scat < 1.0:
                                angle = math.acos(cos_scat)
                                phi = np.random.random() * 2*math.pi
                                up = np.array([0,0,1])
                                if abs(np.dot(dir, up)) > 0.99:
                                    up = np.array([1,0,0])
                                perp = np.cross(dir, up)
                                perp = perp / np.linalg.norm(perp)
                                dir_new = dir * math.cos(angle) + np.cross(perp, dir) * math.sin(angle) + perp * np.dot(perp, dir) * (1 - math.cos(angle))
                            else:
                                dir_new = dir
                            stack.append((e_out, pos, dir_new, False))
                    elif r < p_coh + p_inc + p_ph:
                        # Photoelectric
                        e_out, _, T_e, cos_el = sample_photoelectric(E)
                        events.append({
                            'process': 'photoelectric',
                            'e_in_MeV': E,
                            'e_out_MeV': None,
                            'cos_scatter': None,
                            't_e_MeV': T_e,
                            'cos_electron': cos_el,
                            't_positron_MeV': None
                        })
                        # photon absorbed, no secondary photon
                    else:
                        # Pair production
                        _, _, T_e, cos_el, T_pos = sample_pair(E)
                        events.append({
                            'process': 'pair',
                            'e_in_MeV': E,
                            'e_out_MeV': None,
                            'cos_scatter': None,
                            't_e_MeV': T_e,
                            'cos_electron': cos_el,
                            't_positron_MeV': T_pos
                        })
                        # Photon absorbed. Secondary annihilation photons not generated.
                else:
                    # no interaction, continue loop? Actually we break because photon left the phantom
                    break
                # After interaction, if photon still exists, we continue with next interaction? Actually we already pushed scattered photon onto stack, so we break this loop.
                break
            # end while stack
    return events

# Run simulation
np.random.seed(42)
events = simulate(200000)
print("Number of events collected:", len(events))
# Print first few
for e in events[:5]:
    print(e)