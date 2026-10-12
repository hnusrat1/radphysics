import numpy as np
import math

# Rebuild needed functions in a self-contained way
mec2 = 0.511
r0 = 2.8179403262e-13
r0_sq = r0**2
NA = 6.02214076e23
M_H2O = 18.01528
rho = 1.0
electrons_per_gram_H2O = 10 * NA / M_H2O
electrons_per_cm3 = electrons_per_gram_H2O * rho
F_COMP = 0.95

def sigma_kn(E):
    if E <= 0: return 0
    alpha = E / mec2
    return (2*np.pi*r0_sq * ((1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha) 
                            + np.log(1+2*alpha)/(2*alpha) - (1+3*alpha)/(1+2*alpha)**2))

def mu_comp_rho(E):
    return F_COMP * sigma_kn(E) * electrons_per_gram_H2O

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

def mu_total_rho(E):
    return mu_comp_rho(E) + mu_pe_rho(E) + mu_pair_rho(E) + mu_ray_rho(E)

def mu_linear(E):
    return mu_total_rho(E) * rho

def compton_dsigma_dOmega(E, costheta):
    alpha = E / mec2
    k = 1.0 / (1.0 + alpha * (1.0 - costheta))
    Ep = E * k
    return F_COMP * (r0_sq/2.0) * k**2 * (k + 1.0/k - (1.0 - costheta**2)), Ep

# ===== GEOMETRY =====
SSD = 100.0
depth_poi = 10.0
field_half = 5.0
phantom_half = 15.0
phantom_depth = 30.0
POI = np.array([0.0, 0.0, depth_poi])
source_pos = np.array([0.0, 0.0, -SSD])
E0 = 2.0
mu0 = mu_linear(E0)

# ===== SINGLE SCATTER INTEGRAL =====
# Scattered photon fluence at POI / Phi0:
# Phi_sc/Phi0 = integral over phantom of:
#   (110/r_src)^2 * exp(-mu0 * l1) * n_e * dsigma/dOmega(theta) * exp(-mu(E')*l2) / l2^2 * dV

# r_src = |r - source_pos|, l1 = path length in water from surface to r
# theta = angle between (r - source_pos) and (POI - r)

# For numerical integration, use a 3D grid. The integrand is smooth except near the beam edges.
# Let me use a moderately fine grid.

# Integration grid: 
# z: 0 to 30 cm (phantom depth), spacing 0.5 cm
# x, y: -phantom_half to phantom_half, but need finer near beam
# Use adaptive: grid in beam region, coarser outside

# Let me first define the field boundary: at depth z, half-width = field_half*(SSD+z)/SSD
def field_halfwidth_at_depth(z):
    return field_half * (SSD + z) / SSD

print("Field half-width at depth:")
for z in [0, 5, 10, 15, 20, 30]:
    print(f"  z={z}: {field_halfwidth_at_depth(z):.3f} cm")

# Let me use a structured grid: 
# For z in 0..30, sample every 0.25 cm (121 points)
# For x,y within the beam region, sample every 0.25 cm
# This will be a lot of points but manageable

# Actually, the beam at 30 cm depth has halfwidth 6.5 cm, so the beam region in xy is about 13x13
# Sampling at 0.25 cm spacing: 52x52x121 = 327,000 points
# That should be fast enough.

# But outside the beam, no primary photons, so no first scatter contribution.
# Only integrate over the beam region.

# Let me define the function to compute the integrand

def beam_contains(x, y, z):
    """Check if point (x,y,z) is inside the primary beam pyramid"""
    hw = field_halfwidth_at_depth(z)
    return abs(x) <= hw and abs(y) <= hw

# For each integration point:
# l1 = water path = z (straight down from surface, since beam is aligned with z-axis)
# r_src = sqrt((SSD+z)^2 + x^2 + y^2) = distance from source
# (if x,y are small compared to SSD+z, r_src ≈ SSD+z)

# Actually, the primary photon direction at point (x,y,z) is from source to (x,y,z)
# This direction is nearly parallel to z-axis but has small x,y components
# For the field defined by the pyramid, the photon at (x,y,z) travels along the line
# from source (0,0,-100) to (x,y,z)

# Path length through water: l1 = z / cos(angle from axis) ≈ z * r_src/(SSD+z)
# More precisely: the photon direction unit vector = (x, y, SSD+z)/r_src
# cos(alpha) = (SSD+z)/r_src, so l1 = z/cos(alpha) = z*r_src/(SSD+z)

# But for the primary photons in the beam, they enter the phantom at the point
# (x_entrance, y_entrance, 0) where the line from source through (x,y,z) hits the surface:
# x_entrance = x * SSD/(SSD+z), y_entrance = y * SSD/(SSD+z)
# l1 = length of path inside water = |(x,y,z) - (x_entrance, y_entrance, 0)|
#    = sqrt((x*SSD/(SSD+z))^2 ... )

# For a photon direction u = (x, y, SSD+z)/r_src, the path from surface to z:
# l1 = z / u_z = z * r_src / (SSD+z)

def compute_single_scatter(verbose=True):
    """Compute scattered photon fluence at POI / Phi0 using single-scatter approximation"""
    
    # Define grid
    dz = 0.4
    z_vals = np.arange(0.2, phantom_depth, dz)
    
    Phi_sc = 0.0
    Phi_sc_energy = 0.0  # energy fluence of scattered photons
    
    # Also accumulate spectra and angular distributions
    photon_energy_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
    spectrum_acc = np.zeros(len(photon_energy_edges)-1)
    cotheta_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    angular_acc = np.zeros(len(cotheta_edges)-1)
    
    n_points = 0
    for z in z_vals:
        hw = field_halfwidth_at_depth(z)
        # Sample x,y in the beam region
        nx = max(3, int(2*hw / dz) + 1)
        x_vals = np.linspace(-hw, hw, nx)
        y_vals = np.linspace(-hw, hw, nx)
        
        for x in x_vals:
            for y in y_vals:
                # Only inside the beam pyramid (already constrained by x_vals, y_vals)
                r = np.array([x, y, z])
                
                # Primary fluence at r relative to Phi0
                r_src_dist = np.sqrt(x**2 + y**2 + (SSD+z)**2)
                # Path length in water
                l1 = z * r_src_dist / (SSD + z)
                # Primary photon direction (unit vector)
                u_prim = np.array([x, y, SSD+z]) / r_src_dist
                
                # Attenuation of primary beam
                phi_prim_ratio = (110.0/r_src_dist)**2 * np.exp(-mu0 * l1)
                
                # Direction to POI
                v = POI - r
                l2 = np.linalg.norm(v)
                if l2 < 1e-4:
                    continue
                u_scat = v / l2
                
                # Scattering angle: angle between u_prim and u_scat
                cos_theta_scat = np.dot(u_prim, u_scat)
                if cos_theta_scat > 1: cos_theta_scat = 1
                if cos_theta_scat < -1: cos_theta_scat = -1
                
                # Compton differential cross section
                ds, Ep = compton_dsigma_dOmega(E0, cos_theta_scat)
                
                # Attenuation of scattered photon
                mu_Ep = mu_linear(Ep)
                exp_atten_scat = np.exp(-mu_Ep * l2 / rho)  # l2 is geometric path; /rho to get g/cm^2 (rho=1 anyway)
                # Actually l2 is in cm and mu_linear is cm^-1, so exp(-mu_Ep * l2) directly
                exp_atten_scat = np.exp(-mu_Ep * l2)
                
                # Differential fluence contribution
                dPhi = phi_prim_ratio * electrons_per_cm3 * ds * exp_atten_scat / l2**2 * dz * dz * dz  # dV = dx*dy*dz
                
                # dx, dy vary near edges - since x_vals is uniform, dx = 2hw/(nx-1)
                dx = x_vals[1] - x_vals[0] if len(x_vals) > 1 else 0
                dy = y_vals[1] - y_vals[0] if len(y_vals) > 1 else 0
                # Already multiplied by dz*dz*dz above, need dx*dy*dz
                dPhi = phi_prim_ratio * electrons_per_cm3 * ds * exp_atten_scat / l2**2 * dx * dy * dz
                
                Phi_sc += dPhi
                Phi_sc_energy += dPhi * Ep
                n_points += 1
                
                # Spectrum bin
                idx_e = np.searchsorted(photon_energy_edges, Ep, side='right') - 1
                if 0 <= idx_e < len(spectrum_acc):
                    spectrum_acc[idx_e] += dPhi
                
                # Angular bin
                idx_a = np.searchsorted(cotheta_edges, cos_theta_scat, side='right') - 1
                if 0 <= idx_a < len(angular_acc):
                    angular_acc[idx_a] += dPhi
    
    if verbose:
        print(f"Single-scatter integral done: {n_points} points")
        print(f"Phi_sc/Phi0 = {Phi_sc:.6f}")
        print(f"Energy fluence Phi_sc*E = {Phi_sc_energy:.6f} (relative to Phi0*MeV)")
        print(f"Average scattered photon energy = {Phi_sc_energy/Phi_sc:.4f} MeV")
    
    return Phi_sc, Phi_sc_energy, spectrum_acc, angular_acc

# Run the calculation
Phi_sc, Phi_sc_E, spec, ang = compute_single_scatter()

print(f"\nSpectrum (scattered fluence/Phi0 per bin):")
photon_energy_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
for i in range(len(spec)):
    print(f"  [{photon_energy_edges[i]:.2f}-{photon_energy_edges[i+1]:.2f}]: {spec[i]:.6e}")

print(f"\nAngular distribution (fraction of scattered fluence):")
cotheta_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
ang_norm = ang / np.sum(ang)
for i in range(len(ang)):
    print(f"  [{cotheta_edges[i]:.1f}-{cotheta_edges[i+1]:.1f}]: {ang_norm[i]:.6f}")