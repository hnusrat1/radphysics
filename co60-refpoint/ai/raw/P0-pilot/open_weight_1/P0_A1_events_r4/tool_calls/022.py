import json
from math import sqrt, log, cos, sin, pi
from collections import Counter
import random

# Constants
m_e = 0.510998950
r_e = 2.8179403262e-13
N_A = 6.02214076e23
N_e_per_g = N_A * 10 / 18.01528

# Cross sections
def mu_total_fast(E):
    alpha = E / m_e
    if alpha <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    term1 = (1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - log(1+2*alpha)/alpha)
    term2 = log(1+2*alpha) / (2*alpha)
    term3 = (1+3*alpha) / (1+2*alpha)**2
    sigma = 2*pi * r_e**2 * (term1 + term2 - term3)
    mu_c = sigma * N_e_per_g
    if E < 0.01:
        mu_pe = 4.89 * (E/0.01)**(-3.5)
    elif E < 0.1:
        mu_pe = 4.89 * (E/0.01)**(-3.5)
    elif E < 1.0:
        mu_pe = 0.0017 * (E/0.1)**(-3.2)
    else:
        mu_pe = 7e-6 * (E/1.0)**(-3.5)
    if E <= 2*m_e:
        mu_pp = 0.0
    elif E < 2.0:
        mu_pp = 0.00013 * (E - 2*m_e) / (2.0 - 2*m_e)
    elif E < 5.0:
        mu_pp = 0.00013 + (0.0015-0.00013)*(E-2)/3
    elif E < 10.0:
        mu_pp = 0.0015 + (0.0048-0.0015)*(E-5)/5
    else:
        mu_pp = 0.0048 + (0.0077-0.0048)*(E-10)/10
    if E < 0.01:
        mu_r = 0.16
    elif E < 0.1:
        mu_r = 0.16 * (E/0.01)**(-3.5)
    elif E < 0.5:
        mu_r = 0.0002 * (E/0.1)**(-4)
    else:
        mu_r = 0.0
    mu_t = mu_c + mu_pe + mu_pp + mu_r
    return mu_t, mu_c, mu_pe, mu_pp, mu_r

# Compton sampling
def sample_compton_fast(E, rng):
    alpha = E / m_e
    while True:
        xi1 = rng.random()
        xi2 = rng.random()
        mu = 2.0 * xi1 - 1.0
        ratio = 1.0 / (1.0 + alpha * (1.0 - mu))
        f_kn = ratio**2 * (ratio + 1.0/ratio - (1.0 - mu**2))
        if xi2 <= f_kn * 0.5:
            e_out = E * ratio
            cos_scatter = mu
            break
    t_e = E - e_out
    denom = sqrt(E**2 + e_out**2 - 2.0 * E * e_out * cos_scatter)
    cos_electron = (E - e_out * cos_scatter) / denom if denom > 0 else 1.0
    return e_out, cos_scatter, t_e, cos_electron

# Rotation
def rotate_dir_fast(dx, dy, dz, cos_theta, azimuth):
    if abs(dz) < 0.9:
        norm = sqrt(dx*dx + dy*dy)
        ux = -dy / norm
        uy = dx / norm
        uz = 0.0
    else:
        norm = sqrt(dy*dy + dz*dz)
        ux = 0.0
        u = -dz / norm
        uz = dy / norm
    
    vx = dy*uz - dz*uy
    vy = dz*ux - dx*uz
    vz = dx*uy - dy*ux
    
    sin_theta = sqrt(max(0.0, 1.0 - cos_theta*cos_theta))
    ca = cos(azimuth)
    sa = sin(azimuth)
    
    new_dx = sin_theta * ca * ux + sin_theta * sa * vx + cos_theta * dx
    new_dy = sin_theta * ca * uy + sin_theta * sa * vy + cos_theta * dy
    new_dz = sin_theta * ca * uz + sin_theta * sa * vz + cos_theta * dz
    
    return new_dx, new_dy, new_dz

# Geometry
Z_SURFACE = 100.0
Z_EXIT = 130.0
FIELD_HALF = 5.0
SV_Z_MIN = 109.9
SV_Z_MAX = 110.1
SV_RADIUS_SQ = 1.0