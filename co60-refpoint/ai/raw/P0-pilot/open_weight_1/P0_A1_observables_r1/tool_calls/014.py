import numpy as np
import math

# ===== PHYSICS CONSTANTS =====
mec2 = 0.511  # MeV
r0 = 2.8179403262e-13  # cm
r0_sq = r0**2
NA = 6.02214076e23
M_H2O = 18.01528
rho = 1.0  # g/cm^3
electrons_per_gram = 10 * NA / M_H2O
electrons_per_cm3 = electrons_per_gram * rho
F_COMP = 0.95

# Water composition
# H: Z=1, A=1.008; O: Z=8, A=16.00
# Mass fraction: H=0.1119, O=0.8881
# Atoms per cm^3: N_H = 2*rho*NA/M_H2O, N_O = 1*rho*NA/M_H2O
N_H = 2 * rho * NA / M_H2O
N_O = 1 * rho * NA / M_H2O

# ===== CROSS SECTIONS =====
def sigma_kn(E):
    if E <= 0: return 0
    alpha = E / mec2
    return (2*np.pi*r0_sq * ((1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha) 
                            + np.log(1+2*alpha)/(2*alpha) - (1+3*alpha)/(1+2*alpha)**2))

def mu_comp_rho(E):
    return F_COMP * sigma_kn(E) * electrons_per_gram

def mu_pe_rho(E):
    E = max(float(E), 0.01); E = min(E, 5.0)
    tE = np.array([0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0,3.0,5.0])
    tM = np.array([4.79,1.41,0.519,0.169,0.0770,0.0261,0.0137,0.00480,0.00213,0.000388,0.000111,0.0000235,0.00000881,0.00000412,0.00000212,0.000000838,0.000000443,0.000000220,0.000000123,0.0000000581,0.0000000191,0.00000000392])
    return float(np.exp(np.interp(np.log(E), np.log(tE), np.log(tM))))

def mu_pair_rho(E):
    if E <= 1.022: return 0.0
    E = min(E, 10.0)
    tE = np.array([1.022,1.25,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0,10.0])
    tM = np.array([0.0,0.000639,0.00151,0.00318,0.00496,0.00720,0.0124,0.0169,0.0211,0.0286,0.0340])
    return float(np.exp(np.interp(np.log(E), np.log(tE), np.log(np.maximum(tM,1e-12)))))

def mu_ray_rho(E):
    E = max(float(E), 0.01); E = min(E, 5.0)
    tE = np.array([0.01,0.015,0.02,0.03,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0,3.0,5.0])
    tM = np.array([0.000404,0.000882,0.00154,0.00349,0.00616,0.00716,0.00722,0.00682,0.00574,0.00467,0.00331,0.00250,0.00199,0.00162,0.00116,0.000873,0.000651,0.000502,0.000326,0.000151,0.0000650])
    return float(np.exp(np.interp(np.log(E), np.log(tE), np.log(tM))))

def mu_total(E):
    """Total linear attenuation coefficient (cm^-1)"""
    return (mu_comp_rho(E) + mu_pe_rho(E) + mu_pair_rho(E) + mu_ray_rho(E)) * rho

def get_mu_components(E):
    """Return (mu_comp, mu_pe, mu_pair, mu_ray) all linear (cm^-1)"""
    return (mu_comp_rho(E)*rho, mu_pe_rho(E)*rho, mu_pair_rho(E)*rho, mu_ray_rho(E)*rho)

# ===== COMPTON SCATTERING SAMPLING =====
def sample_compton(E_inc, n_samples=100000):
    """Sample Compton scattering angles from Klein-Nishina.
    Returns (cos_theta, E_scattered) arrays.
    Uses inverse CDF sampling."""
    alpha = E_inc / mec2
    
    # Generate a large number of cos(theta) values
    # Use rejection sampling or inverse CDF
    # The KN differential cross section: dσ/dΩ ∝ k²(k + 1/k - sin²θ)
    # where k = 1/(1+α(1-cosθ))
    
    # CDF approach: sample cosθ via numerical inverse CDF
    # Tabulate the CDF
    ct_grid = np.linspace(-1, 1, 2001)
    k = 1.0 / (1.0 + alpha*(1.0 - ct_grid))
    dsigma = 2*np.pi * (r0_sq/2.0) * k**2 * (k + 1/k - (1 - ct_grid**2))
    cdf = np.cumsum(dsigma)
    cdf = cdf / cdf[-1]
    
    # Sample u ~ U(0,1), find ct where cdf = u
    u = np.random.rand(n_samples)
    ct = np.interp(u, cdf, ct_grid)
    
    # Compute scattered energy
    E_sc = E_inc / (1.0 + alpha*(1.0 - ct))
    return ct, E_sc

def sample_compton_single(E_inc):
    """Sample one Compton scattering event. Returns (cos_theta, E_scattered)"""
    ct, E_sc = sample_compton(E_inc, 1)
    return ct[0], E_sc[0]

# ===== GEOMETRY =====
SSD = 100.0
depth_poi = 10.0
field_half = 5.0
phantom_half = 15.0
phantom_depth = 30.0
POI = np.array([0.0, 0.0, depth_poi])

# Scoring cylinder: radius 1 cm, half-thickness 0.1 cm, centered at (0,0,10)
cyl_radius = 1.0
cyl_half_z = 0.1
cyl_z_min = depth_poi - cyl_half_z
cyl_z_max = depth_poi + cyl_half_z
cyl_volume = np.pi * cyl_radius**2 * 2*cyl_half_z
print(f"Cylinder volume: {cyl_volume:.4f} cm^3")

def is_in_cylinder(x, y, z):
    """Check if point is in scoring cylinder"""
    return (cyl_z_min <= z <= cyl_z_max) and (x**2 + y**2 <= cyl_radius**2)

def ray_cylinder_intersection(p, u):
    """Compute the track length of a ray (point p, direction u) through the cylinder.
    Returns track length or 0."""
    # Parametrize: p + t*u, t >= 0
    # Cylinder: x^2 + y^2 <= 1, z in [z_min, z_max]
    
    # z-boundary crossings
    tz_enter = None
    tz_exit = None
    if abs(u[2]) > 1e-10:
        t1 = (cyl_z_min - p[2]) / u[2]
        t2 = (cyl_z_max - p[2]) / u[2]
        tz_enter = min(t1, t2)
        tz_exit = max(t1, t2)
        if tz_exit < 0:
            return 0.0
        if tz_enter < 0:
            tz_enter = 0.0
    
    # Cylinder side crossings
    a = u[0]**2 + u[1]**2
    b = 2*(p[0]*u[0] + p[1]*u[1])
    c = p[0]**2 + p[1]**2 - cyl_radius**2
    
    t_side_enter = None
    t_side_exit = None
    if a > 1e-12:
        disc = b**2 - 4*a*c
        if disc >= 0:
            sqrt_disc = math.sqrt(disc)
            t1 = (-b - sqrt_disc) / (2*a)
            t2 = (-b + sqrt_disc) / (2*a)
            if t2 < 0:
                return 0.0
            t_side_enter = max(0.0, t1)
            t_side_exit = t2
        else:
            return 0.0
    else:
        # direction is along z-axis, no side crossing possible
        pass
    
    # Find the entry and exit points
    candidates_enter = []
    candidates_exit = []
    if tz_enter is not None:
        z_at_t = p[2] + tz_enter*u[2]
        x_at_t = p[0] + tz_enter*u[0]
        y_at_t = p[1] + tz_enter*u[1]
        if x_at_t**2 + y_at_t**2 <= cyl_radius**2 + 1e-9:
            candidates_enter.append(tz_enter)
    if t_side_enter is not None:
        z_at_t = p[2] + t_side_enter*u[2]
        if cyl_z_min <= z_at_t <= cyl_z_max:
            candidates_enter.append(t_side_enter)
    
    if tz_exit is not None:
        z_at_t = p[2] + tz_exit*u[2]
        x_at_t = p[0] + tz_exit*u[0]
        y_at_t = p[1] + tz_exit*u[1]
        if x_at_t**2 + y_at_t**2 <= cyl_radius**2 + 1e-9:
            candidates_exit.append(tz_exit)
    if t_side_exit is not None:
        z_at_t = p[2] + t_side_exit*u[2]
        if cyl_z_min <= z_at_t <= cyl_z_max:
            candidates_exit.append(t_side_exit)
    
    if not candidates_enter or not candidates_exit:
        # Check if starting point is inside
        if is_in_cylinder(p[0], p[1], p[2]):
            t_enter = 0.0
        else:
            return 0.0
    else:
        t_enter = min(candidates_enter)
    
    t_exit = min(candidates_exit) if candidates_exit else 1e9
    
    if t_exit < t_enter:
        return 0.0
    
    track = t_exit - t_enter
    return max(0.0, track)

def move_photon_to_phantom(p, u):
    """Move photon from source position along direction u until it reaches z=0 plane (phantom surface).
    Returns position at surface, or None if it misses the phantom."""
    # Photon starts at source (0,0,-SSD)
    # Move along u until z=0
    if u[2] <= 0:
        return None
    t = SSD / u[2]
    x = p[0] + t*u[0]
    y = p[1] + t*u[1]
    if -phantom_half <= x <= phantom_half and -phantom_half <= y <= phantom_half:
        return np.array([x, y, 0.0])
    return None

# Test cylinder intersection
p_test = np.array([0.0, 0.0, 9.0])
u_test = np.array([0.0, 0.0, 1.0])
L = ray_cylinder_intersection(p_test, u_test)
print(f"Test track length through cylinder (axial): {L:.4f} (expect 0.2)")
p_test2 = np.array([0.0, 0.0, 9.9])
u_test2 = np.array([0.5, 0.0, 0.866])
L2 = ray_cylinder_intersection(p_test2, u_test2)
print(f"Test track length (angled): {L2:.4f}")