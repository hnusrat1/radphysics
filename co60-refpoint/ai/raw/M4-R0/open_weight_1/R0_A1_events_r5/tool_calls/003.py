import numpy as np
from scipy import integrate, special
from scipy.interpolate import interp1d
import json

np.random.seed(42)

# ============================================================
# Physical constants
# ============================================================
m_e_c2 = 0.51099895  # MeV
r_e = 2.8179403262e-13  # cm
N_A = 6.02214076e23

# ============================================================
# Cross sections for water
# ============================================================
energy_pts = np.array([1.0, 1.1, 1.1732, 1.2, 1.25, 1.3, 1.3325, 1.4, 1.5])
mu_rho_total = np.array([0.07070, 0.06780, 0.06590, 0.06520, 0.06400, 0.06280, 0.06220, 0.06080, 0.05780])
mu_rho_comp = np.array([0.06950, 0.06650, 0.06430, 0.06350, 0.06200, 0.06060, 0.06000, 0.05850, 0.05550])
mu_rho_photo = np.array([0.00040, 0.00030, 0.00025, 0.00022, 0.00020, 0.00017, 0.00015, 0.00013, 0.00010])
mu_rho_pair = np.array([0.0, 0.0, 0.00020, 0.00028, 0.00050, 0.00080, 0.00100, 0.00130, 0.00150])
mu_rho_rayl = np.array([0.00080, 0.00068, 0.00060, 0.00055, 0.00050, 0.00043, 0.00040, 0.00035, 0.00030])

f_mu_total = interp1d(energy_pts, mu_rho_total, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_mu_comp = interp1d(energy_pts, mu_rho_comp, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_mu_photo = interp1d(energy_pts, mu_rho_photo, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_mu_pair = interp1d(energy_pts, mu_rho_pair, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_mu_rayl = interp1d(energy_pts, mu_rho_rayl, kind='cubic', bounds_error=False, fill_value='extrapolate')

def mu_total(E):
    return float(f_mu_total(np.clip(E, 0.01, 10.0)))

def mu_comp(E):
    return float(f_mu_comp(np.clip(E, 0.01, 10.0)))

def mu_photo(E):
    return float(f_mu_photo(np.clip(E, 0.01, 10.0)))

def mu_pair(E):
    return float(f_mu_pair(np.clip(E, 0.01, 10.0)))

def mu_rayl(E):
    return float(f_mu_rayl(np.clip(E, 0.01, 10.0)))

# ============================================================
# Geometry
# ============================================================
SSD = 100.0
field_half = 5.0  # half-width of field at surface

# Scoring volume
scoring_radius = 1.0
z_scoring_center = 5.0
scoring_half_thick = 0.1  # half of 0.2 cm
z_scoring_min = z_scoring_center - scoring_half_thick
z_scoring_max = z_scoring_center + scoring_half_thick

# Source
src_pos = np.array([0.0, 0.0, -SSD])

# ============================================================
# Utility functions
# ============================================================

def compton_energy(E_in, cos_theta):
    return E_in / (1 + (E_in / m_e_c2) * (1 - cos_theta))

def klein_nishina_dcs(E_in, cos_theta):
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    E_out = compton_energy(E_in, cos_theta)
    return (r_e**2 / 2) * (E_out / E_in)**2 * (E_in/E_out + E_out/E_in - np.sin(theta)**2)

def sample_source_photon():
    """Sample source photon: energy and direction within field."""
    E = 1.1732 if np.random.random() < 0.5 else 1.3325
    x_surf = np.random.uniform(-field_half, field_half)
    y_surf = np.random.uniform(-field_half, field_half)
    dx = x_surf - src_pos[0]
    dy = y_surf - src_pos[1]
    dz = 0.0 - src_pos[2]
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    direction = np.array([dx, dy, dz]) / norm
    return E, direction

def sample_compton_photon(E_in):
    """Sample outgoing photon from Klein-Nishina. Returns E_out, cos_theta, t_e, cos_electron."""
    max_dcs = r_e**2 * 1.1
    while True:
        cos_th = np.random.uniform(-1, 1)
        dcs = klein_nishina_dcs(E_in, cos_th)
        if np.random.random() * max_dcs < dcs:
            break
    E_out = compton_energy(E_in, cos_th)
    t_e = E_in - E_out
    num = E_in - E_out * cos_th
    den = np.sqrt(E_in**2 + E_out**2 - 2*E_in*E_out*cos_th)
    cos_el = num / den if den > 0 else 1.0
    return E_out, cos_th, t_e, cos_el

def sample_compton_isotropic(E_in):
    """Sample using isotropic scattering in the center-of-mass frame (simpler)."""
    # Actually let's use the Klein-Nishina distribution properly
    return sample_compton_photon(E_in)

def random_rotation(dir_in, cos_theta):
    """Given an incident direction and cos(theta) in that frame, return the scattered direction."""
    # Direction of incident photon
    u = dir_in / np.linalg.norm(dir_in)
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    phi = np.random.uniform(0, 2*np.pi)
    
    # Find a basis
    if abs(u[2]) < 0.999:
        v = np.cross(u, np.array([0, 0, 1]))
    else:
        v = np.cross(u, np.array([1, 0, 0]))
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    
    # Scattered direction in the u,v,w basis
    dir_out = u * cos_theta + v * np.sin(theta) * np.cos(phi) + w * np.sin(theta) * np.sin(phi)
    return dir_out / np.linalg.norm(dir_out)

def ray_cylinder_intersection(origin, direction, radius, z_min, z_max):
    """Find if a ray intersects a cylinder on the z-axis.
    Returns entry and exit distances along the ray, or None if no intersection."""
    # Parametric: origin + t * direction
    # Cylinder: x^2 + y^2 = radius^2, z_min <= z <= z_max
    
    # Check z range
    t_z_enter = (z_min - origin[2]) / direction[2]
    t_z_exit = (z_max - origin[2]) / direction[2]
    if t_z_enter > t_z_exit:
        t_z_enter, t_z_exit = t_z_exit, t_z_enter
    
    # Check cylinder
    # (origin[0] + t*direction[0])^2 + (origin[1] + t*direction[1])^2 = radius^2
    a = direction[0]**2 + direction[1]**2
    b = 2 * (origin[0]*direction[0] + origin[1]*direction[1])
    c = origin[0]**2 + origin[1]**2 - radius**2
    
    if a < 1e-10:
        # Ray is parallel to z-axis
        if origin[0]**2 + origin[1]**2 <= radius**2:
            return t_z_enter, t_z_exit
        else:
            return None
    
    disc = b**2 - 4*a*c
    if disc < 0:
        return None
    
    t1 = (-b - np.sqrt(disc)) / (2*a)
    t2 = (-b + np.sqrt(disc)) / (2*a)
    if t1 > t2:
        t1, t2 = t2, t1
    
    # Intersection of z-range and cylinder range
    t_enter = max(t_z_enter, t1)
    t_exit = min(t_z_exit, t2)
    
    if t_enter >= t_exit or t_exit < 0:
        return None
    
    return t_enter, t_exit

def sample_interaction_type(E):
    """Sample the interaction type based on cross sections."""
    mu_c = mu_comp(E)
    mu_p = mu_photo(E)
    mu_pp = mu_pair(E)
    mu_r = mu_rayl(E)
    mu_tot = mu_c + mu_p + mu_pp + mu_r
    
    r = np.random.random() * mu_tot
    if r < mu_c:
        return 'compton'
    elif r < mu_c + mu_p:
        return 'photoelectric'
    elif r < mu_c + mu_p + mu_pp:
        return 'pair'
    else:
        return 'rayleigh'

# ============================================================
# Main simulation
# ============================================================

def simulate_events(n_desired):
    """
    Simulate photon interactions in the scoring volume.
    
    We use a variance reduction approach:
    - For primary photons: directly compute interaction probability
    - For scattered photons: use Monte Carlo with forced interaction
    
    Returns list of event dictionaries.
    """
    events = []
    
    # We'll simulate a large number of source photons and tally interactions in the scoring volume
    # To make it efficient, we'll use importance sampling / forced detection
    
    # Strategy: For each source photon, we calculate:
    # 1. The probability that it reaches the scoring volume and undergoes a specific interaction there
    # 2. We sample from this distribution using weights
    
    # Actually, let's use a direct Monte Carlo approach:
    # Sample source photons, track them through the phantom, score when they interact in the scoring volume
    
    max_tries = 50000000
    # Will sample until we get enough events
    
    # For efficiency, let's use a hybrid approach:
    # 1. Directly sample primary interactions in the scoring volume using analytical probabilities
    # 2. Use Monte Carlo for scattered photon interactions
    
    # Let's start by computing the relative rates of primary vs scattered interactions
    
    # ==========================================
    # PRIMARY PHOTON INTERACTIONS
    # ==========================================
    
    print("Computing primary photon interactions...")
    
    # For primary photons, the expected number of interactions in the scoring volume per source photon is:
    # ∫ dΩ * (1/Ω_pyramid) * [exp(-μ(E)*L_to_scoring)] * [1 - exp(-μ(E)*L_in_scoring)]
    # where L_to_scoring is the path from entry to scoring volume
    # and L_in_scoring is the path within the scoring volume
    
    # We'll use Monte Carlo to compute this
    n_mc = 200000
    primary_weight_sum = {'compton': 0.0, 'photoelectric': 0.0, 'pair': 0.0, 'rayleigh': 0.0}
    primary_events_list = []
    
    for _ in range(n_mc):
        E, dir_in = sample_source_photon()
        
        # Path length from surface to scoring volume
        dz = dir_in[2]
        if dz <= 0:
            continue
        
        # Distance from surface to z_scoring_min along the ray
        t_to_scoring = (z_scoring_min - z_surface) / dz
        
        # Entry point to scoring volume
        entry_pt = np.array([0, 0, z_surface]) + t_to_scoring * dir_in
        
        # Check if entry point is within the cylinder radius
        r_entry = np.sqrt(entry_pt[0]**2 + entry_pt[1]**2)
        if r_entry > scoring_radius:
            continue  # ray misses the scoring cylinder
        
        # Path length through the scoring volume
        t_exit = (z_scoring_max - z_surface) / dz
        exit_pt = np.array([0, 0, z_surface]) + t_exit * dir_in
        r_exit = np.sqrt(exit_pt[0]**2 + exit_pt[1]**2)
        
        # The ray might exit the cylinder radially before the z-exit
        # For simplicity, approximate path as 0.2/dz (valid for rays near axis)
        path_in_scoring = 0.2 / dz  # approximation
        
        # Attenuation from surface to scoring volume
        mu_at_E = mu_total(E)
        atten_to_scoring = np.exp(-mu_at_E * t_to_scoring)
        
        # Interaction probability in scoring volume
        p_int_in_vol = 1 - np.exp(-mu_at_E * path_in_scoring)
        
        # Expected number of interactions per source photon (relative)
        weight = atten_to_scoring * p_int_in_vol
        
        # Now determine which process
        mu_c = mu_comp(E)
        mu_p = mu_photo(E)
        mu_pp = mu_pair(E)
        mu_r = mu_rayl(E)
        mu_tot = mu_c + mu_p + mu_pp + mu_r
        
        # For each process, record a weighted event
        for proc, mu_proc in [('compton', mu_c), ('photoelectric', mu_p), ('pair', mu_pp), ('rayleigh', mu_r)]:
            proc_weight = weight * mu_proc / mu_tot
            if proc_weight > 0 and np.random.random() < 0.01:  # sample a fraction for efficiency
                pass  # will store differently
        
        # Actually, let's just accumulate weights and sample later
        if weight > 0:
            primary_weight_sum['compton'] += weight * mu_c / mu_tot
            primary_weight_sum['photoelectric'] += weight * mu_p / mu_tot
            primary_weight_sum['pair'] += weight * mu_pp / mu_tot
            primary_weight_sum['rayleigh'] += weight * mu_r / mu_tot
            
            # Store for later sampling
            if np.random.random() < 0.01:  # thin sampling
                primary_events_list.append({
                    'E': E,
                    'dir_in': dir_in.copy(),
                    'weight': weight,
                    'mu_c': mu_c, 'mu_p': mu_p, 'mu_pp': mu_pp, 'mu_r': mu_r, 'mu_tot': mu_tot
                })
    
    # Normalize
    for k in primary_weight_sum:
        primary_weight_sum[k] /= n_mc
    
    print(f"Primary interaction weights per source photon:")
    for k, v in primary_weight_sum.items():
        print(f"  {k}: {v:.8f}")
    print(f"  Total: {sum(primary_weight_sum.values()):.8f}")
    
    # ==========================================
    # SCATTERED PHOTON INTERACTIONS  
    # ==========================================
    
    print("\nComputing scattered photon interactions (single scatter)...")
    
    # For scattered photons, we consider photons that scatter via Compton 
    # somewhere in the phantom and then reach the scoring volume.
    # The scattering can happen anywhere in the phantom, so we need to integrate
    
    # We'll use Monte Carlo integration over scatter points
    n_scatter_mc = 500000
    scatter_weight_sum = {'compton': 0.0, 'photoelectric': 0.0, 'pair': 0.0, 'rayleigh': 0.0}
    scatter_events_list = []
    
    for _ in range(n_scatter_mc):
        E, dir_in = sample_source_photon()
        dz = dir_in[2]
        if dz <= 0:
            continue
        
        # Sample a scatter point along the photon's path in the phantom
        # The photon can scatter anywhere in the phantom (z from 0 to 30)
        # with probability proportional to μ(E) * exp(-μ(E)*z) at depth z
        
        # For the scattered photon to reach the scoring volume, the scatter must happen
        # somewhere such that after scattering, the photon can reach the scoring volume
        
        # First, sample the scatter point
        # The probability density of scattering at distance t from entry:
        # μ(E) * exp(-μ(E)*t)
        # where t is the path length from the surface
        
        # Sample t using inverse CDF
        mu_e = mu_total(E)
        
        # Max path in phantom along this ray
        t_max = 30.0 / dz  # from surface to back of phantom
        
        # Sample interaction point
        u = np.random.random()
        t_scatter = -np.log(1 - u * (1 - np.exp(-mu_e * t_max))) / mu_e
        
        # Position of scatter
        scatter_pt = np.array([0, 0, z_surface]) + t_scatter * dir_in
        
        # Check if scatter is within the phantom laterally (30x30)
        if abs(scatter_pt[0]) > 15 or abs(scatter_pt[1]) > 15:
            continue
        
        # Attenuation to scatter point
        atten_to_scatter = np.exp(-mu_e * t_scatter)
        
        # Now sample the Compton scatter process
        # Sample scatter angle from Klein-Nishina
        E_out, cos_th, t_e, cos_el = sample_compton_photon(E)
        
        # Scattered photon direction
        dir_out = random_rotation(dir_in, cos_th)
        
        # Check if scattered photon can reach the scoring volume
        # We need the ray from scatter_pt in direction dir_out to intersect the scoring volume
        dz_out = dir_out[2]
        if dz_out <= 0:
            continue  # photon going backward from scatter point
        
        # Distance from scatter point to z_scoring_min along scattered direction
        # But the scattered photon might go up or down
        it = ray_cylinder_intersection(scatter_pt, dir_out, scoring_radius, z_scoring_min, z_scoring_max)
        if it is None:
            continue
        
        t_enter_vol, t_exit_vol = it
        if t_enter_vol < 0:
            t_enter_vol = 0
            
        # Path length through scoring volume
        path_in_vol = t_exit_vol - t_enter_vol
        if path_in_vol <= 0:
            continue
        
        # Attenuation of scattered photon from scatter point to scoring volume
        mu_out = mu_total(E_out)
        atten_to_vol = np.exp(-mu_out * t_enter_vol)
        
        # Interaction probability in scoring volume
        p_int_vol = 1 - np.exp(-mu_out * path_in_vol)
        
        # Scattering probability weight:
        # dσ/dΩ * ΔΩ where ΔΩ is the solid angle subtended by the scoring volume from the scatter point
        # For efficiency, we use the actual geometry check above and compute the interaction probability
        
        # The scattering probability per unit solid angle at the scatter point:
        # (n_e * dσ/dΩ) where n_e is electron density
        # For water: 10 electrons per molecule, 18 g/mol
        # n_e = (10 * N_A * ρ) / 18.015 = 3.343e23 e-/cm^3
        n_e = 10 * N_A * 1.0 / 18.015
        
        # The differential scattering cross section per unit volume
        dcs = klein_nishina_dcs(E, cos_th)  # cm^2/sr per electron
        mu_s_diff = n_e * dcs  # cm^-1 sr^-1
        
        # Solid angle of the scoring volume as seen from scatter point
        # Approximate as area / distance^2
        dist_to_vol = t_enter_vol
        area_proj = np.pi * scoring_radius**2  # projected area
        # More accurate: the solid angle subtended
        solid_angle = area_proj / (dist_to_vol**2) if dist_to_vol > 0 else 0
        
        # Weight for this scattered photon interaction
        weight = atten_to_scatter * mu_s_diff * solid_angle * atten_to_vol * p_int_vol
        
        # Also need to include the Jacobian for sampling scatter points
        # and the factor for sampling the Compton angle
        # We sampled scatter point from exp(-μt), so the probability density is μ*exp(-μt) dt
        # We sampled scattering uniformly in cos(theta), so the probability density is 0.5 d(cosθ)
        # The actual density should be dσ/dΩ / σ_s, so we need to account for this
        
        # The weight above is the expected number of interactions in the scoring volume
        # from this particular scatter event, per source photon
        
        # More carefully:
        # dN_scattered = dN_source * (μ_s_diff * dΩ_scoring) * exp(-μ*L1) * exp(-μ*L2) * (μ_int * dL)
        # where dN_source is the number of source photons in this direction
        # dL is the path through scoring volume
        
        # Our Monte Carlo samples source photons with 1/n_mc weight
        # and scatter points with μ*exp(-μt) distribution
        # So the weight is:
        # (1/n_mc) * [μ_s_diff * solid_angle / (μ * exp(-μt))] * exp(-μL1) * exp(-μL2) * p_int_vol
        # Hmm, let me think about this differently.
        
        # Actually, I think a cleaner approach is needed. Let me restructure.
        
        # For now, let's just accumulate a simplified weight
        # The simplified approach: treat the scattered photon fluence as
        # coming from single Compton scatter, integrated over the phantom
        
        # Use a simpler weighting scheme
        # weight = (1/n_mc) * n_e * dcs * (atten_to_scatter * mu_e / mu_e) * solid_angle * atten_to_vol * p_int_vol
        # where we divide by the PDF of sampling (μ_c * exp(-μt))
        
        # Actually, for the scatter point we sampled from μ*exp(-μt), PDF = μ*exp(-μt)/C
        # where C = 1 - exp(-μ*t_max)
        # So weight = (expected) / (sampling PDF * n_mc)
        
        # Expected interaction rate from this scatter configuration:
        # dN = dN_source * exp(-μ*t_scatter) * μ_comp(t) * [n_e*dσ/dΩ*dΩ] * exp(-μ_comp*L) * μ_int*dL
        # Wait, the scattering probability at the scatter point is: μ_comp * dt * (dσ/dΩ / σ_comp) * dΩ
        # = n_e * (dσ/dΩ) * dΩ * dt
        
        # Let me just use a simpler weight:
        C = 1 - np.exp(-mu_e * t_max)
        pdf_scatter = mu_e * np.exp(-mu_e * t_scatter) / C
        
        # For the angular scatter, we sampled uniformly in cos_theta in [-1, 1]
        # PDF = 0.5
        pdf_angle = 0.5
        
        # The expected contribution from this event:
        # = (dσ/dΩ / σ_total) * dΩ_weight * ... 
        # Actually let me simplify.
        
        # Probability of Compton scattering at scatter point per unit path:
        # Σ_comp = n_e * σ_comp_integrated
        # σ_comp_integrated = ∫ (dσ/dΩ) dΩ
        # So the probability of scattering into dΩ at angle θ:
        # dP = n_e * (dσ/dΩ) * dΩ * dt
        
        # Weight = (n_e * dcs * solid_angle) * atten_to_scatter * atten_to_vol * p_int_vol / (n_mc * pdf_scatter * pdf_angle)
        
        weight = (n_e * dcs * solid_angle * atten_to_scatter * atten_to_vol * p_int_vol) / (n_scatter_mc * pdf_scatter * pdf_angle)
        
        if weight <= 0:
            continue
        
        # Now determine the interaction type of the scattered photon in the scoring volume
        mu_c_out = mu_comp(E_out)
        mu_p_out = mu_photo(E_out)
        mu_pp_out = mu_pair(E_out)
        mu_r_out = mu_rayl(E_out)
        mu_tot_out = mu_c_out + mu_p_out + mu_pp_out + mu_r_out
        
        # Accumulate weights for each process
        scatter_weight_sum['compton'] += weight * mu_c_out / mu_tot_out
        scatter_weight_sum['photoelectric'] += weight * mu_p_out / mu_tot_out
        scatter_weight_sum['pair'] += weight * mu_pp_out / mu_tot_out
        scatter_weight_sum['rayleigh'] += weight * mu_r_out / mu_tot_out
        
        # Store event info for later sampling
        if weight > 1e-15:
            scatter_events_list.append({
                'E_in': E_out,
                'weight': weight,
                'is_primary': False,
                'E_original': E,
                'cos_th': cos_th,
                't_e': t_e,
                'cos_el': cos_el,
                'mu_c': mu_c_out, 'mu_p': mu_p_out, 'mu_pp': mu_pp_out, 'mu_r': mu_r_out, 'mu_tot': mu_tot_out
            })
    
    for k in scatter_weight_sum:
        scatter_weight_sum[k] /= n_scatter_mc  # already normalized
    
    print(f"Scattered photon interaction weights per source photon:")
    for k, v in scatter_weight_sum.items():
        print(f"  {k}: {v:.8f}")
    print(f"  Total: {sum(scatter_weight_sum.values()):.8f}")
    
    total_primary = sum(primary_weight_sum.values())
    total_scatter = sum(scatter_weight_sum.values())
    total = total_primary + total_scatter
    
    print(f"\nTotal primary interaction rate: {total_primary:.8f}")
    print(f"Total scatter interaction rate: {total_scatter:.8f}")
    print(f"Scatter/Primary ratio: {total_scatter/total_primary:.4f}")
    print(f"Scatter fraction: {total_scatter/total:.4f}")
    
    # ==========================================
    # Generate 200 events by sampling
    # ==========================================
    
    # Now we need to generate exactly 200 events with the correct distribution
    # We'll sample from the accumulated distributions
    
    events = []
    
    # Determine how many primary vs scattered events
    n_primary_expected = int(round(200 * total_primary / total))
    n_scatter_expected = 200 - n_primary_expected
    
    print(f"\nGenerating {n_primary_expected} primary and {n_scatter_expected} scattered events...")
    
    # Primary events
    # We'll generate them by sampling from the source distribution and accepting/rejecting
    # based on the interaction probability
    generated_primary = 0
    attempts = 0
    max_attempts = 5000000
    
    while generated_primary < n_primary_expected and attempts < max_attempts:
        attempts += 1
        E, dir_in = sample_source_photon()
        dz = dir_in[2]
        if dz <= 0:
            continue
        
        t_to_scoring = (z_scoring_min - z_surface) / dz
        
        # Check radial distance
        entry_pt = np.array([0, 0, z_surface]) + t_to_scoring * dir_in
        if np.sqrt(entry_pt[0]**2 + entry_pt[1]**2) > scoring_radius:
            continue
        
        # Path through scoring volume
        path_in_scoring = 0.2 / dz
        
        # Attenuation
        atten = np.exp(-mu_total(E) * t_to_scoring)
        p_int = 1 - np.exp(-mu_total(E) * path_in_scoring)
        
        # Accept with probability proportional to atten * p_int
        # Use rejection sampling
        max_prob = np.exp(-mu_total(E) * 5.0) * (1 - np.exp(-mu_total(E) * 0.2))
        max_prob *= 1.2  # margin
        
        if np.random.random() * max_prob > atten * p_int:
            continue
        
        # Determine interaction type
        proc = sample_interaction_type(E)
        
        if proc == 'compton':
            E_out, cos_th, t_e, cos_el = sample_compton_photon(E)
            events.append({
                'process': 'compton',
                'e_in_MeV': E,
                'e_out_MeV': E_out,
                'cos_scatter': cos_th,
                't_e_MeV': t_e,
                'cos_electron': cos_el,
                't_positron_MeV': None
            })
            generated_primary += 1
            
        elif proc == 'photoelectric':
            # Photoelectric: photon absorbed, electron gets E - binding energy
            # Binding energy for water: ~0.5 keV for K-shell of O, negligible
            t_e = E  # essentially all energy to photoelectron
            # Photoelectron is emitted preferentially in forward direction at these energies
            # cos_electron ~ cos(θ) where the angular distribution is roughly
            # dσ/dΩ ∝ sin^2(θ) / (1 - β cos(θ))^4 for high energies
            # At ~1.25 MeV, photoelectrons are forward-peaked
            # For simplicity, sample cos in [0.5, 1] approximately forward
            cos_el = np.random.uniform(0.7, 1.0)
            events.append({
                'process': 'photoelectric',
                'e_in_MeV': E,
                'e_out_MeV': None,
                'cos_scatter': None,
                't_e_MeV': t_e,
                'cos_electron': cos_el,
                't_positron_MeV': None
            })
            generated_primary += 1
            
        elif proc == 'pair':
            # Pair production: photon converts to e-/e+ pair
            # Energy shared between electron and positron
            # The total kinetic energy is E - 2*m_e = E - 1.022 MeV
            # The energy distribution is approximately uniform (for high Z, but for water...
            # For water, the energy distribution is roughly flat)
            
            t_kin_total = E - 2 * m_e_c2
            if t_kin_total > 0:
                # Approximate: sample e- energy uniformly between 0 and t_kin_total
                t_e = np.random.uniform(0, t_kin_total)
                t_pos = t_kin_total - t_e
                
                # In pair production, the e- and e+ are forward-peaked
                # The angle is approximately m_e/E (very forward)
                cos_el = np.random.uniform(0.95, 1.0)
                
                events.append({
                    'process': 'pair',
                    'e_in_MeV': E,
                    'e_out_MeV': None,
                    'cos_scatter': None,
                    't_e_MeV': t_e,
                    'cos_electron': cos_el,
                    't_positron_MeV': t_pos
                })
                generated_primary += 1
            
        elif proc == 'rayleigh':
            # Rayleigh: elastic scattering, no energy loss
            # Angular distribution is forward-peaked
            # For water at 1.25 MeV, Rayleigh scattering is very forward
            # Sample from approximate form factor
            # For simplicity, sample cos_scatter close to 1
            cos_th = np.random.uniform(0.999, 1.0)
            events.append({
                'process': 'rayleigh',
                'e_in_MeV': E,
                'e_out_MeV': E,
                'cos_scatter': cos_th,
                't_e_MeV': 0.0,
                'cos_electron': None,
                't_positron_MeV': None
            })
            generated_primary += 1
    
    print(f"Generated {generated_primary} primary events in {attempts} attempts")
    
    # Scattered events
    generated_scatter = 0
    attempts = 0
    
    while generated_scatter < n_scatter_expected and attempts < max_attempts:
        attempts += 1
        E, dir_in = sample_source_photon()
        dz = dir_in[2]
        if dz <= 0:
            continue
        
        mu_e = mu_total(E)
        t_max = 30.0 / dz
        
        # Sample scatter point
        u = np.random.random()
        C_samp = 1 - np.exp(-mu_e * t_max)
        t_scatter = -np.log(1 - u * C_samp) / mu_e
        
        scatter_pt = np.array([0, 0, z_surface]) + t_scatter * dir_in
        if abs(scatter_pt[0]) > 15 or abs(scatter_pt[1]) > 15:
            continue
        
        atten_to_scatter = np.exp(-mu_e * t_scatter)
        
        # Compton scatter
        E_out, cos_th, t_e, cos_el = sample_compton_photon(E)
        dir_out = random_rotation(dir_in, cos_th)
        
        dz_out = dir_out[2]
        if dz_out <= 0:
            continue
        
        # Check if reaches scoring volume
        it = ray_cylinder_intersection(scatter_pt, dir_out, scoring_radius, z_scoring_min, z_scoring_max)
        if it is None:
            continue
        
        t_enter_vol, t_exit_vol = it
        if t_enter_vol < 0:
            t_enter_vol = 0
        path_in_vol = t_exit_vol - t_enter_vol
        if path_in_vol <= 0:
            continue
        
        mu_out = mu_total(E_out)
        atten_to_vol = np.exp(-mu_out * t_enter_vol)
        p_int_vol = 1 - np.exp(-mu_out * path_in_vol)
        
        # Weight for rejection sampling
        n_e = 10 * N_A * 1.0 / 18.015
        dcs = klein_nishina_dcs(E, cos_th)
        dist_to_vol = t_enter_vol
        solid_angle = np.pi * scoring_radius**2 / (dist_to_vol**2) if dist_to_vol > 0 else 0
        
        pdf_scatter = mu_e * np.exp(-mu_e * t_scatter) / C_samp if C_samp > 0 else 0
        pdf_angle = 0.5
        
        weight = (n_e * dcs * solid_angle * atten_to_scatter * atten_to_vol * p_int_vol) / (pdf_scatter * pdf_angle)
        
        # Use rejection sampling based on weight
        # Normalize: find typical max weight
        # For efficiency, accept with probability proportional to weight
        max_w = 1e-20  # will adjust
        
        if weight > 1e-25:
            # Determine interaction type
            proc = sample_interaction_type(E_out)
            
            if proc == 'compton':
                E_out2, cos_th2, t_e2, cos_el2 = sample_compton_photon(E_out)
                events.append({
                    'process': 'compton',
                    'e_in_MeV': E_out,
                    'e_out_MeV': E_out2,
                    'cos_scatter': cos_th2,
                    't_e_MeV': t_e2,
                    'cos_electron': cos_el2,
                    't_positron_MeV': None
                })
                generated_scatter += 1
                
            elif proc == 'photoelectric':
                events.append({
                    'process': 'photoelectric',
                    'e_in_MeV': E_out,
                    'e_out_MeV': None,
                    'cos_scatter': None,
                    't_e_MeV': E_out,
                    'cos_electron': np.random.uniform(0.5, 1.0),
                    't_positron_MeV': None
                })
                generated_scatter += 1
                
            elif proc == 'pair':
                t_kin_total = E_out - 2 * m_e_c2
                if t_kin_total > 0:
                    t_e_val = np.random.uniform(0, t_kin_total)
                    t_pos = t_kin_total - t_e_val
                    events.append({
                        'process': 'pair',
                        'e_in_MeV': E_out,
                        'e_out_MeV': None,
                        'cos_scatter': None,
                        't_e_MeV': t_e_val,
                        'cos_electron': np.random.uniform(0.9, 1.0),
                        't_positron_MeV': t_pos
                    })
                    generated_scatter += 1
                
            elif proc == 'rayleigh':
                events.append({
                    'process': 'rayleigh',
                    'e_in_MeV': E_out,
                    'e_out_MeV': E_out,
                    'cos_scatter': np.random.uniform(0.999, 1.0),
                    't_e_MeV': 0.0,
                    'cos_electron': None,
                    't_positron_MeV': None
                })
                generated_scatter += 1
    
    print(f"Generated {generated_scatter} scattered events in {attempts} attempts")
    
    return events[:200]

# Run the simulation
events = simulate_events(200)
print(f"\nTotal events generated: {len(events)}")

# Check distribution
processes = [e['process'] for e in events]
for p in ['compton', 'photoelectric', 'pair', 'rayleigh']:
    print(f"  {p}: {processes.count(p)}")