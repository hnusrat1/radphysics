import numpy as np
import math
import random

# Constants
m_e = 0.51099895
r_e_sq = 7.9408e-26
n_e = 3.3428e23

# Cross section formulas (same as before)
def mu_Compton(E):
    alpha = E / m_e
    t1 = (1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - math.log(1+2*alpha)/alpha)
    t2 = math.log(1+2*alpha) / (2*alpha)
    t3 = (1+3*alpha) / (1+2*alpha)**2
    sigma = 2*math.pi * r_e_sq * (t1 + t2 - t3)
    return n_e * sigma

def mu_Rayleigh(E):
    if E <= 0:
        return 0.0
    return 0.0008 * (1.0 / max(E, 0.01))**2

def mu_Photoelectric(E):
    if E <= 0:
        return 0.0
    return 1.9e-5 * (1.0 / max(E, 0.01))**3.5

def mu_Pair(E):
    if E <= 1.022:
        return 0.0
    if E < 2.0:
        return 0.0004 * (E - 1.022) / (2.0 - 1.022)
    elif E < 3.0:
        return 0.0004 + (0.0007 - 0.0004) * (E - 2.0)
    elif E < 5.0:
        return 0.0007 + (0.0020 - 0.0007) * (E - 3.0) / 2.0
    else:
        return 0.0020 * (E / 5.0)**0.5

def mu_Total(E):
    return mu_Compton(E) + mu_Rayleigh(E) + mu_Photoelectric(E) + mu_Pair(E)

# Klein-Nishina differential cross section (cm^2 / sr per electron)
def KN_differential(alpha, cos_theta):
    """dσ/dΩ (cm^2/sr) per electron for Compton scattering."""
    x = 1 - cos_theta
    if x < 1e-12:
        x = 1e-12
    e_out_ratio = 1.0 / (1 + alpha * x)
    ds_dOmega = (r_e_sq / 2.0) * e_out_ratio**2 * (e_out_ratio + 1.0/e_out_ratio - (1 - cos_theta**2))
    return ds_dOmega  # cm^2/sr

# Total Compton cross section per electron (cm^2)
def sigma_KN(alpha):
    t1 = (1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - math.log(1+2*alpha)/alpha)
    t2 = math.log(1+2*alpha) / (2*alpha)
    t3 = (1+3*alpha) / (1+2*alpha)**2
    sigma = 2*math.pi * r_e_sq * (t1 + t2 - t3)
    return sigma

# Primary fluence at depth (relative to Phi0)
def primary_fluence(z):
    """Return primary fluence at depth z (cm) relative to Phi0."""
    mu = mu_Total(2.0)
    return math.exp(-mu * z)

# Single-scatter contribution to fluence spectrum at scoring point
# We'll use Monte Carlo integration over scatter locations

def compute_scatter_spectrum(num_samples=200000):
    """
    Compute the fluence spectrum of singly-scattered photons at the scoring point
    (0,0,10) relative to Phi0.
    
    We sample scatter points within the phantom volume that are within the field.
    Returns: list of (energy, weight) pairs.
    """
    E_in = 2.0
    alpha_in = E_in / m_e
    mu_Comp_in = mu_Compton(E_in)
    sigma_Comp_in = sigma_KN(alpha_in)  # per electron
    n_e = 3.3428e23
    
    # Scoring point
    x0, y0, z0 = 0.0, 0.0, 10.0
    
    # Phantom limits
    x_half = 15.0
    y_half = 15.0
    z_min = 0.0
    z_max = 30.0
    
    # Field half-size at any depth z
    # Source at z=-100, field half-size at surface = 5 cm
    # At depth z, field half-size = 5 * (100 + z) / 100 = 5 + 0.05*z
    def field_half(z):
        return 5.0 + 0.05 * z
    
    results = []
    
    for i in range(num_samples):
        # Sample scatter point uniformly in the phantom
        # But we can importance sample: scatter points closer to the scoring point contribute more.
        # Let's simply sample uniformly within the field region.
        z_s = random.uniform(z_min, z_max)
        fh = field_half(z_s)
        x_s = random.uniform(-fh, fh)
        y_s = random.uniform(-fh, fh)
        
        # Check if inside phantom
        if abs(x_s) > x_half or abs(y_s) > y_half:
            continue
        
        # Direction from scatter point to scoring point
        dx = x0 - x_s
        dy = y0 - y_s
        dz = z0 - z_s
        R = math.sqrt(dx*dx + dy*dy + dz*dz)
        if R < 0.01:
            continue
        
        # Incident direction is along +z (approximation)
        cos_theta = dz / R  # scattering angle cosine
        
        # Scattered photon energy
        e_out = E_in / (1 + alpha_in * (1 - cos_theta))
        
        # Differential cross section for scattering into the direction of scoring point
        ds_dOmega = KN_differential(alpha_in, cos_theta)  # per electron, cm^2/sr
        
        # Probability that a scatter at this point results in a photon heading to scoring point
        # The solid angle subtended by the scoring point at the scatter point is dΩ = area / R^2
        # But we are integrating over scatter points, so the contribution to fluence at scoring point
        # is: Φ_prim(z_s) * μ_Comp * dV * (1/σ_KN) * (dσ/dΩ) * (1/R^2) * attenuation_scat
        # where dV = dx_s dy_s dz_s.
        
        # Primary fluence at scatter point
        phi_prim = primary_fluence(z_s)  # relative to Phi0
        
        # Attenuation of scattered photon from scatter point to scoring point
        # The path goes through water. We need to compute the distance through water.
        # Since the phantom is large, the path from scatter point to scoring point is entirely in water
        # if the line segment is inside the phantom. We'll compute the fraction that is in water.
        # For simplicity, assume the line is entirely in water (most cases).
        # Check if the line segment from (x_s,y_s,z_s) to (0,0,10) stays within phantom.
        # For a conservative estimate, we'll assume full path is in water (since both points are inside).
        mu_scat = mu_Total(e_out)
        atten = math.exp(-mu_scat * R)
        
        # Volume element: we are sampling uniformly over the field region.
        # The volume of the sampling region is V_total = (2*x_half)*(2*y_half)*(z_max-z_min) but we 
        # limit x,y to field size. The actual volume sampled is:
        # For each z, we sample from -fh to fh in x and y, so dz * (2*fh)^2.
        # But we are sampling points uniformly in the entire phantom volume (not restricted to field).
        # Actually we sample within field. The total volume of the field pyramid within phantom.
        # Since we are using uniform sampling, we need to weight by the inverse of the sampling density.
        # The sampling density is 1 / (volume of region).
        # The region is: for each z, x in [-fh, fh], y in [-fh, fh], z in [0,30].
        # The volume V_sample = ∫_{0}^{30} (2*fh)^2 dz = ∫ 4*(5+0.05z)^2 dz
        # But we also clip to phantom boundaries. Let's just compute the effective volume.
        
        # For simplicity, we'll use a constant sampling density over the whole field pyramid.
        # But we'll compute the weight as 1 / (number of samples per unit volume) * (contributions).
        
        # Actually, simpler: we can sample uniformly over the entire phantom volume and use the 
        # probability of being in the field as a weight. But let's just compute the contribution
        # relative to Phi0.
        
        # The contribution to fluence at scoring point from a single scatter event in dV is:
        # dΦ_scat = Φ_prim(z_s) * μ_Comp * dV * (dσ/dΩ)/σ_Comp * (1/R^2) * atten
        # where dV = dx_s dy_s dz_s is the volume element.
        
        # In our Monte Carlo integration, we sample N points uniformly in a volume V_total.
        # The estimate of the total scatter fluence is:
        # Φ_scat_total = (V_total / N) * Σ Φ_prim(z_s) * μ_Comp * (dσ/dΩ)/σ_Comp * (1/R^2) * atten
        
        # V_total: the volume of the field region in the phantom.
        # Compute V_total by integration.
        
        # Let's compute V_total later. For now, store samples.
        
        # For each sample, compute contribution weight
        w = phi_prim * mu_Comp_in * (ds_dOmega / sigma_Comp_in) * (1.0 / R**2) * atten
        
        results.append((e_out, w))
    
    # Now compute total volume
    # V_total = ∫_{0}^{30} (2*fh)^2 dz, but limited by phantom boundaries.
    # fh = 5 + 0.05*z, so (2*fh)^2 = 4*(5+0.05z)^2
    # Integrate from z=0 to z=30:
    V_total = 0.0
    for z in np.linspace(0, 30, 1001):
        fh = 5.0 + 0.05*z
        dz = 30.0/1000
        V_total += 4 * fh**2 * dz
    
    # Scale weights
    scaling = V_total / num_samples
    results = [(e, w * scaling) for e, w in results]
    
    return results

# Test
print("Computing scatter spectrum...")
random.seed(42)
scatter_data = compute_scatter_spectrum(50000)
print(f"Got {len(scatter_data)} samples")
# Check total scatter fluence
total_scat_fluence = sum(w for _, w in scatter_data)
print(f"Total scatter fluence = {total_scat_fluence:.4f} * Phi0")
# Primary fluence at scoring point
phi_prim = primary_fluence(10.0)
print(f"Primary fluence = {phi_prim:.4f} * Phi0")
print(f"Scatter-to-primary ratio = {total_scat_fluence/phi_prim:.4f}")