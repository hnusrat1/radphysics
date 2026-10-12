import numpy as np
from scipy import integrate, special
import json

# Physical constants
m_e_c2 = 0.51099895  # MeV
r_e = 2.8179403262e-13  # cm
N_A = 6.02214076e23  # mol^-1

# ============================================================
# Cross sections for water
# ============================================================

# Mass attenuation coefficients (cm^2/g) - from NIST data
# For water at key energies
energy_pts = np.array([1.0, 1.1, 1.1732, 1.2, 1.25, 1.3, 1.3325, 1.4, 1.5])

# Total
mu_rho_total = np.array([0.07070, 0.06780, 0.06590, 0.06520, 0.06400, 0.06280, 0.06220, 0.06080, 0.05780])
# Compton
mu_rho_comp = np.array([0.06950, 0.06650, 0.06430, 0.06350, 0.06200, 0.06060, 0.06000, 0.05850, 0.05550])
# Photoelectric
mu_rho_photo = np.array([0.00040, 0.00030, 0.00025, 0.00022, 0.00020, 0.00017, 0.00015, 0.00013, 0.00010])
# Pair production
mu_rho_pair = np.array([0.0, 0.0, 0.00020, 0.00028, 0.00050, 0.00080, 0.00100, 0.00130, 0.00150])
# Rayleigh
mu_rho_rayl = np.array([0.00080, 0.00068, 0.00060, 0.00055, 0.00050, 0.00043, 0.00040, 0.00035, 0.00030])

from scipy.interpolate import interp1d
f_mu_total = interp1d(energy_pts, mu_rho_total, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_mu_comp = interp1d(energy_pts, mu_rho_comp, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_mu_photo = interp1d(energy_pts, mu_rho_photo, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_mu_pair = interp1d(energy_pts, mu_rho_pair, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_mu_rayl = interp1d(energy_pts, mu_rho_rayl, kind='cubic', bounds_error=False, fill_value='extrapolate')

def mu_total(E):
    return f_mu_total(E)  # cm^-1 (since density=1 g/cm^3)

def mu_comp(E):
    return f_mu_comp(E)

def mu_photo(E):
    return f_mu_photo(E)

def mu_pair(E):
    return f_mu_pair(E)

def mu_rayl(E):
    return f_mu_rayl(E)

# ============================================================
# Geometry
# ============================================================
SSD = 100.0  # cm
z_surface = 0.0
z_scoring_center = 5.0  # cm depth
scoring_radius = 1.0  # cm
scoring_half_thick = 0.1  # cm (total thickness 0.2 cm)
z_scoring_min = z_scoring_center - scoring_half_thick
z_scoring_max = z_scoring_center + scoring_half_thick

# Source position
src_pos = np.array([0.0, 0.0, -SSD])  # (x, y, z)

# Field half-size at surface
field_half = 5.0  # cm

# Angular half-widths
theta_x_max = np.arctan(field_half / SSD)  # ~0.04996 rad
theta_y_max = np.arctan(field_half / SSD)

# ============================================================
# Source sampling
# ============================================================

def sample_source_photon():
    """Sample a source photon: energy and direction."""
    # Energy: equal probability for the two lines
    E = 1.1732 if np.random.random() < 0.5 else 1.3325
    
    # Direction: uniform in solid angle within the pyramid
    # Sample direction cosines using the pyramid's angular bounds
    # The pyramid is defined by the field at the phantom surface
    # For a point at (x, y, z=0) in the field, direction from source is (x, y, 100)/sqrt(x^2+y^2+100^2)
    # Uniform in solid angle means uniform in the projection onto the sphere
    # We can sample uniformly on the field at z=0 and project
    
    x_surf = np.random.uniform(-field_half, field_half)
    y_surf = np.random.uniform(-field_half, field_half)
    
    # Direction vector from source to surface point
    dx = x_surf - src_pos[0]
    dy = y_surf - src_pos[1]
    dz = z_surface - src_pos[2]  # = 100
    
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    direction = np.array([dx, dy, dz]) / norm
    
    return E, direction

# ============================================================
# Compton kinematics
# ============================================================

def compton_energy(E_in, cos_theta):
    """Outgoing photon energy."""
    return E_in / (1 + (E_in / m_e_c2) * (1 - cos_theta))

def klein_nishina_dcs(E_in, cos_theta):
    """dσ/dΩ in cm^2/sr per electron."""
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    E_out = compton_energy(E_in, cos_theta)
    return (r_e**2 / 2) * (E_out / E_in)**2 * (E_in/E_out + E_out/E_in - np.sin(theta)**2)

def sample_compton(E_in):
    """Sample Compton scatter angle from Klein-Nishina."""
    # Use rejection sampling with uniform cos(theta) proposal
    # Maximum of dσ/dΩ is at cos(theta)=1
    
    # Find the maximum of the DCS over cos(theta) in [-1, 1]
    # It's at cos=1, value = r_e^2
    max_dcs = r_e**2 * 1.1  # slight margin
    
    while True:
        cos_th = np.random.uniform(-1, 1)
        dcs = klein_nishina_dcs(E_in, cos_th)
        if np.random.random() * max_dcs < dcs:
            break
    
    E_out = compton_energy(E_in, cos_th)
    
    # Electron kinematics
    t_e = E_in - E_out
    # Electron angle from momentum conservation
    # cos(θ_e) = (E_in - E_out*cos_th) / sqrt(E_in^2 + E_out^2 - 2*E_in*E_out*cos_th)
    num = E_in - E_out * cos_th
    den = np.sqrt(E_in**2 + E_out**2 - 2*E_in*E_out*cos_th)
    cos_el = num / den if den > 0 else 1.0
    
    return E_out, cos_th, t_e, cos_el

# ============================================================
# Calculate interaction probability in scoring volume for primary photons
# ============================================================

def path_length_in_scoring(direction):
    """Calculate the path length of a photon through the scoring volume.
    The scoring volume is a cylinder on the beam axis, radius 1.0 cm, 
    from z=4.9 to z=5.1 cm.
    """
    # Ray from some starting point (at z=z_scoring_min entrance) along direction
    # We need to find the intersection of the ray with the cylinder
    
    dz_dir = direction[2]
    if dz_dir <= 0:
        return 0.0  # photon not moving downward
    
    # Entry at z = z_scoring_min
    # Position at entry
    # We need to find if the ray enters the cylinder
    
    # Actually, let's parameterize the trajectory
    # A point on the ray: (x0, y0, z0) + t * (dx, dy, dz)
    # where z0 is the entry point into the scoring volume (z=z_scoring_min or z=z_scoring_max)
    
    # For the ray to be in the scoring volume, z must be between z_scoring_min and z_scoring_max
    # and sqrt(x^2 + y^2) <= scoring_radius
    
    # Let's compute the intersection times with the two planes
    t_entry_z = (z_scoring_min - z_surface) / dz_dir  # entry to the slab
    t_exit_z = (z_scoring_max - z_surface) / dz_dir   # exit from the slab
    
    # Check if the ray intersects the cylinder (radius scoring_radius)
    # Parametric: (x0 + t*dx)^2 + (y0 + t*dy)^2 = scoring_radius^2
    # where x0, y0 is position at z_surface (entry to phantom)
    
    # But we don't know the exact entry point for a given ray...
    # Actually, for a photon that reaches the scoring volume, its trajectory goes through
    # the cylinder axis. Let's just use the straight-line distance through the slab
    # and check the radial distance.
    
    return None  # placeholder, will compute differently

# Let me take a different approach - Monte Carlo integration

def compute_primary_interaction_rate():
    """
    Compute the relative interaction rate for primary photons in the scoring volume.
    Returns a list of (process, E_in, E_out, cos_scatter, t_e, cos_electron, t_positron) 
    for each event, with appropriate weights.
    """
    # For primary photons, the interaction rate in the scoring volume is:
    # R = ∫ dΩ ∫ dE * source_intensity(E, Ω) * exp(-μ(E)*L_to_scoring) * (1 - exp(-μ(E)*L_in_scoring))
    # where L_to_scoring is the path length from surface to scoring volume entry
    # and L_in_scoring is the path length within the scoring volume
    
    # Since the scoring volume is on the beam axis and small, we can approximate:
    # - Photons traveling near the beam axis
    # - L_to_scoring ≈ 5 / cos(θ) where θ is the angle from the beam axis
    # - L_in_scoring ≈ 0.2 / cos(θ)
    
    # For a more accurate calculation, let's do Monte Carlo integration
    
    return None

# Let me compute things more precisely
print("Testing geometry and cross sections...")

# Test cross sections
for E in [1.1732, 1.3325]:
    mu_t = mu_total(E)
    mu_c = mu_comp(E)
    mu_p = mu_photo(E)
    mu_pp = mu_pair(E)
    mu_r = mu_rayl(E)
    print(f"E={E} MeV: total={mu_t:.6f}, comp={mu_c:.6f}, photo={mu_p:.6f}, pair={mu_pp:.6f}, rayl={mu_r:.6f}")

# Test source sampling
n_test = 10000
directions = []
energies = []
for _ in range(n_test):
    E, d = sample_source_photon()
    directions.append(d)
    energies.append(E)

directions = np.array(directions)
print(f"Mean cos(theta) of source: {np.mean(directions[:, 2]):.6f}")
print(f"Mean |dx|: {np.mean(np.abs(directions[:, 0])):.6f}")
print(f"Energy proportions: 1.1732={energies.count(1.1732)/n_test:.4f}, 1.3325={energies.count(1.3325)/n_test:.4f}")

# Path length to scoring volume for central ray
central_dir = np.array([0, 0, 1])
path_to_scoring_center = 5.0 / central_dir[2]
print(f"Path length to scoring center for central ray: {path_to_scoring_center} cm")

# Attenuation for primary photons at scoring volume
for E in [1.1732, 1.3325]:
    att = np.exp(-mu_total(E) * 5.0)
    print(f"Attenuation after 5 cm water at E={E} MeV: {att:.4f}")

# Probability of interaction in 0.2 cm slab
for E in [1.1732, 1.3325]:
    p_int = 1 - np.exp(-mu_total(E) * 0.2)
    print(f"Interaction prob in 0.2 cm at E={E} MeV: {p_int:.6f}")
    # By process
    p_c = 1 - np.exp(-mu_comp(E) * 0.2)
    p_ph = 1 - np.exp(-mu_photo(E) * 0.2)
    p_pp = 1 - np.exp(-mu_pair(E) * 0.2)
    p_r = 1 - np.exp(-mu_rayl(E) * 0.2)
    print(f"  Comp={p_c:.6f}, Photo={p_ph:.6f}, Pair={p_pp:.6f}, Rayl={p_r:.6f}")
    # Relative proportions
    total_p = p_c + p_ph + p_pp + p_r
    print(f"  Comp={p_c/total_p:.4f}, Photo={p_ph/total_p:.4f}, Pair={p_pp/total_p:.4f}, Rayl={p_r/total_p:.4f}")