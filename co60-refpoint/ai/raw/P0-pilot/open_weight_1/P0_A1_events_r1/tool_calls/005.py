import numpy as np
import math
from scipy.integrate import quad

# Let me compute things more carefully using analytical methods

# Constants
m_e = 0.511  # MeV
r_e = 2.818e-13  # cm
N_A = 6.022e23
molar_mass_H2O = 18.015
electrons_per_cm3 = N_A * 10 / molar_mass_H2O * 1.0  # ~3.34e23

# Source geometry
SSD = 100.0  # cm
E0 = 2.0  # MeV
field_half = 5.0  # cm at surface

# Point of interest: 10 cm depth, on axis
d = 10.0  # cm
source_to_poi = SSD + d  # 110 cm

# Beam cross-section at depth d
field_at_depth = field_half * (SSD + d) / SSD  # half-width at depth
print(f"Beam half-width at 10 cm depth: {field_at_depth:.2f} cm")
print(f"Beam area at 10 cm depth: {(2*field_at_depth)**2:.1f} cm^2")

# Scoring volume
scoring_r = 1.0  # cm
scoring_thick = 0.2  # cm
scoring_center_z = 10.0  # cm
scoring_area = np.pi * scoring_r**2
scoring_vol = scoring_area * scoring_thick
print(f"Scoring volume: area={scoring_area:.4f} cm^2, volume={scoring_vol:.4f} cm^3")

# Fraction of beam area covered by scoring volume
area_fraction = scoring_area / (2*field_at_depth)**2
print(f"Area fraction: {area_fraction:.4f}")

#=============================================================================
# Cross sections
#=============================================================================

def compton_cs_per_electron(E):
    if E <= 0:
        return 0
    epsilon = E / m_e
    k = epsilon
    sigma = 2 * np.pi * r_e**2 * (
        (1 + k) / k**2 * (2 * (1 + k) / (1 + 2*k) - np.log(1 + 2*k) / k) +
        np.log(1 + 2*k) / (2*k) - (1 + 3*k) / (1 + 2*k)**2
    )
    return sigma

def mu_comp(E):
    if E <= 0:
        return 0
    return compton_cs_per_electron(E) * electrons_per_cm3

def mu_photo(E):
    if E <= 0:
        return 0
    if E < 0.01:
        return 10.0 * (0.01/E)**3.2
    elif E < 0.1:
        return 10.0 * (0.01/E)**2.9
    elif E < 0.5:
        return 0.04 * (0.1/E)**2.7
    else:
        return 0.0004 * (0.5/E)**3.0

def mu_pair(E):
    if E < 1.022:
        return 0.0
    if E < 3.0:
        return 0.0078 * (E - 1.022) / (2.0 - 1.022)
    else:
        return 0.012 * (1 + 0.15 * np.log(E/3))

def mu_rayleigh(E):
    if E <= 0:
        return 0
    if E < 0.01:
        return 8.0 * (0.01/E)**2.6
    elif E < 0.1:
        return 8.0 * (0.01/E)**2.3
    elif E < 1.0:
        return 0.06 * (0.1/E)**2.3
    else:
        return 0.0003 * (1.0/E)**2.5

def mu_total(E):
    return mu_comp(E) + mu_photo(E) + mu_pair(E) + mu_rayleigh(E)

#=============================================================================
# Primary fluence calculation
#=============================================================================

# The primary photon fluence at the scoring volume (with phantom present) is:
# Φ_prim = Φ_0 * exp(-μ_total * d) 
# where Φ_0 is the primary fluence at the point with phantom removed.

# But the scoring volume is small and on the central axis. The primary fluence
# is uniform across the beam cross-section at depth.

# At the scoring volume (within the beam), the primary photon fluence:
# Φ_prim(d, r=0) = Φ_0 * exp(-μ_total * d)

# Where Φ_0 = (source particles per unit solid angle into the beam) / (distance)^2
# For a uniform source within the beam pyramid:
# The fluence on axis at distance r from source is:
# Φ_0 = N / (A_beam_at_r) where N is number of photons emitted into the beam and
# A_beam_at_r is the beam area at distance r.

# Actually, for a point source emitting uniformly into solid angle Ω,
# the fluence at distance r on axis is (N/Ω) * (Ω_spot/A_spot) = N / (4π r^2) for isotropic,
# but here we have a finite beam. 
# The number of photons crossing the beam area at distance r is just N (all particles).
# So the fluence (track length per unit volume) at any point on axis within the beam
# is related to the number of photons per unit area crossing a perpendicular plane.

# Let's just compute the primary interaction rate in the scoring volume.

# Number of primary photons crossing a unit area perpendicular to the beam axis
# at depth d, per unit source strength:
# φ_prim(d) = exp(-μ*d) * (number of source photons in beam) / (beam area at depth d)

# But we need relative proportion, not absolute. Let's use Phi0 normalization.

# Actually, the problem states:
# Normalization: Phi0 is the primary photon fluence at the location of the point of interest,
# 110 cm from the source, with the phantom removed.

# So Phi0 = N_beam / A_beam(110 cm) where N_beam is the number of photons emitted 
# by the source into the beam.

# With phantom present, the primary fluence at the POI is:
# Phi_prim = Phi0 * exp(-μ*total * 10 cm)

# For scattered fluence, we need to compute the scattered photon contribution.

# Let me use a simpler approach: let's use known scatter-to-primary ratios
# and approximate the scattered energy distribution.

# For 2 MV photons, 10x10 cm field, 10 cm depth:
# The scatter-to-primary ratio (SPR) for fluence is approximately 0.3-0.6
# (depends on definition). Let me estimate.

# Actually, let me compute it using a first-scatter approximation.
# I'll compute the contribution from first Compton scatter.

print("\n=== Analytical primary fluence calculation ===")

mu_at_E0 = mu_total(E0)
print(f"Total μ at {E0} MeV: {mu_at_E0:.6f} cm^-1")
print(f"Primary transmission to 10 cm: {np.exp(-mu_at_E0*10):.4f}")

# Primary fluence at scoring volume relative to Phi0:
Phi_prim_rel = np.exp(-mu_at_E0 * 10)  # relative to Phi0
print(f"Primary fluence (relative to Phi0): {Phi_prim_rel:.4f}")

#=============================================================================
# First scatter contribution
#=============================================================================

# For the first-scatter contribution to the fluence at the scoring volume,
# we need to consider Compton scattering from various points in the phantom.

# Let me compute the single-scatter contribution analytically by integrating
# over the beam.

# At each point in the phantom where a primary photon scatters, the scattered
# photon may reach the scoring volume.

# This is a complex 3D integral. Let me try a simpler approach.

# Let me use the "convolution" method for scatter.

print("\n=== First-scatter estimation ===")

# For a rough estimate, let me compute the scatter contribution from a few
# representative points.

# The first-scatter contribution to fluence at the scoring volume comes from
# primary photons that Compton-scatter at some point and then reach the scoring
# volume without further interaction.

# Let me integrate over the beam volume:

# Fluence contribution from scatter at point (r', z') to scoring volume:
# dΦ_scat = Φ0 * exp(-μ*|z') * (μ_comp * dz') * [Klein-Nishina cross section at angle θ]
# * exp(-μ*|path from scatter to scoring vol) * (1/|r_scatter_to_scoring|^2) * acceptance

# This is complex but doable with numerical integration.

# Let me sample from the first-scatter distribution instead.

# For the first scatter distribution that reaches the scoring volume:
# 1. Sample an interaction point uniformly along the primary beam
# 2. Sample Compton scatter angle
# 3. Check if scattered photon reaches scoring volume
# 4. Weight by attenuation

# Let me use Monte Carlo integration for the first scatter contribution.

def compute_first_scatter_fluence(num_samples=100000):
    """Compute first scatter contribution to fluence at scoring volume via MC"""
    total_weight = 0.0  # weight = contribution to fluence / Phi0
    
    for _ in range(num_samples):
        # Sample interaction point
        # z' from 0 to depth where primary beam exists
        # Weight by attenuation from source to interaction point
        
        # Primary photon survival probability depends on z'
        # p(z') = μ_total(E0) * exp(-μ_total(E0) * z')
        # Sample z' from the distribution of interaction points
        u = np.random.random()
        z_prime = -np.log(1 - u) / mu_total(E0)
        
        if z_prime > 30:  # beyond phantom
            continue
        
        # x', y' position within beam at depth z'
        half_width = field_half * (SSD + z_prime) / SSD
        x_prime = np.random.uniform(-half_width, half_width)
        y_prime = np.random.uniform(-half_width, half_width)
        
        # Check if within phantom
        if abs(x_prime) > 15 or abs(y_prime) > 15:
            continue
        
        # Now sample Compton scatter
        # Sample cosθ from Klein-Nishina
        k = E0 / m_e
        
        # Use importance sampling - scatter towards scoring volume
        # Direction from scatter point to scoring volume center
        dx_sc = 0 - x_prime
        dy_sc = 0 - y_prime
        dz_sc = scoring_center_z - z_prime
        r_sc = math.sqrt(dx_sc**2 + dy_sc**2 + dz_sc**2)
        
        if r_sc == 0:
            continue
        
        cos_theta_sc = dz_sc / r_sc  # cosine of angle between incident photon direction (z) and scattered direction to scoring vol
        
        # For a photon moving along z, the scattering angle to reach (x',y') at scoring center
        # is the angle between (0,0,1) and (dx_sc, dy_sc, dz_sc)/r_sc
        cos_theta_required = dz_sc / r_sc
        
        # The probability of this exact angle is zero in a continuous distribution,
        # so we need to weight by the Klein-Nishina differential cross section.
        
        # Actually, for computing fluence, we need to integrate over all scatter directions
        # that reach the scoring volume. The scoring volume has a finite size.
        
        # For simplicity, let me use a different approach:
        # Sample a Compton scatter angle from the full distribution,
        # then propagate the scattered photon towards the scoring volume
        # and check if it enters it.
        
        cos_theta = np.random.uniform(-1, 1)  # will use rejection later
        
        # Let me use rejection sampling for KN
        # ... this is getting complex
        
    return total_weight / num_samples

# Let me try a completely different approach
# I'll compute the scatter contribution using known empirical data

# For 2 MeV photons in water, 10x10 cm field, 10 cm depth:
# The scatter fraction for dose is about 20-30% for 10x10 at 10 cm depth for Co-60
# For fluence, the scatter component is larger

# Let me use the following approach:
# - Primary fluence: from exponential attenuation
# - Scatter fluence: use a simple model where the scatter fluence energy 
#   distribution is approximately uniform in some range

# Actually, let me just use a variance-reduced Monte Carlo simulation
# where I force interactions in the scoring volume.

print("\n=== Building a more efficient simulation ===")

# Let me use the following approach:
# 1. For primary photons: calculate the probability that a primary photon 
#    interacts in the scoring volume using analytical formulas
# 2. For scattered photons: use a first-scatter approximation with
#    importance sampling

# The probability for a primary photon to interact in the scoring volume:
# P_prim = exp(-μ_total * d) * (1 - exp(-μ_total * scoring_thick)) * (scoring_area / beam_area_at_depth)
# where d = 10 cm (to center of scoring volume)

P_prim_interact = (np.exp(-mu_at_E0 * (scoring_center_z - scoring_thick/2)) * 
                   (1 - np.exp(-mu_at_E0 * scoring_thick)) * 
                   (scoring_area / (2*field_at_depth)**2))
print(f"P(primary photon interacts in scoring vol): {P_prim_interact:.6f}")

# Number of primary photons needed for 200 events: 200 / P_prim_interact / (primary fraction)
# Let me estimate primary fraction in interactions
# At 2.0 MeV, primary interaction probability in scoring vol is as above
# For scattered photons, the interaction probability depends on their energy

# Let me try a more direct approach. I'll use a simplified but complete
# Monte Carlo where I only track photons that might reach the scoring volume.

# Actually, let me just use the fact that we need 200 events and compute 
# everything analytically, sampling from the known distributions.

# I'll determine the energy distribution of photons entering the scoring volume,
# then sample interactions.

print("\n=== Computing photon fluence spectrum at scoring volume ===")

# The photon fluence at the scoring volume consists of:
# 1. Primary (unscattered) photons: energy = 2.0 MeV
# 2. First-Compton-scattered photons from various locations
# 3. Multiple-scattered photons (higher orders)
# 4. Annihilation photons from pair production (0.511 MeV)

# For a reasonable approximation, I'll compute the first-scatter contribution
# to the fluence spectrum using numerical integration

# Let me compute the first-scatter contribution to the fluence at the scoring volume

def dsigma_comp_dOmega(E_in, costheta):
    """Klein-Nishina differential cross section per electron (cm^2/sr)"""
    k = E_in / m_e
    kp = k / (1 + k * (1 - costheta))
    if kp <= 0:
        return 0
    return (r_e**2 / 2) * (kp/k)**2 * (k/kp + kp/k - 1 + costheta**2)

def scatter_to_fluence(E0, scoring_pos, scatter_pos):
    """
    Compute contribution to fluence at scoring position from a single scatter
    at scatter_pos. Returns (fluence_weight, scattered_energy).
    """
    # Vector from scatter point to scoring volume center
    r_vec = scoring_pos - scatter_pos
    r = np.linalg.norm(r_vec)
    if r == 0:
        return 0, 0
    
    # Direction from scatter to scoring
    dir_to_scoring = r_vec / r
    
    # Incident photon direction is along +z
    # Scattering angle between incident direction (0,0,1) and direction to scoring
    costheta = dir_to_scoring[2]  # z-component of direction to scoring
    
    # Energy after Compton scatter
    E_scat = E0 / (1 + (E0/m_e) * (1 - costheta))
    
    if E_scat <= 0:
        return 0, 0
    
    # Differential cross section for this angle (per electron)
    ds_dOmega = dsigma_comp_dOmega(E0, costheta)
    
    # Number of electrons per cm^3 at scatter point
    ne = electrons_per_cm3
    
    # Attenuation from source to scatter point
    # Primary photon travels from source to scatter point along +z
    d_source_to_scatter = scatter_pos[2]  # depth
    atten_source_to_scatter = np.exp(-mu_total(E0) * d_source_to_scatter)
    
    # Attenuation of scattered photon from scatter point to scoring volume
    d_scatter_to_scoring = r
    # The scattered photon energy is E_scat
    atten_scatter_to_scoring = np.exp(-mu_total(E_scat) * d_scatter_to_scoring)
    
    # The scattering probability per unit volume at scatter point:
    # dP/dV = ne * ds_dOmega * (steradians subtended by scoring vol at scatter point) 
    #         * atten_source * atten_scatter * (1/r^2)
    
    # Solid angle subtended by scoring volume from scatter point
    # Approximate: scoring area perpendicular to direction / r^2
    scoring_area_proj = scoring_area * abs(dir_to_scoring[2])  # area projected perpendicular to direction
    # Actually, the scoring volume is a cylinder. The solid angle is approximately
    # the area of the cylinder's cross-section perpendicular to the viewing direction.
    
    # For simplicity, use the flat area of the cylinder end cap projected
    dOmega = scoring_area * abs(dir_to_scoring[2]) / r**2  # approximate
    
    # Fluence contribution: number of scattered photons per unit area at scoring volume
    # fluence = (ne * ds_dOmega * dV * dOmega * atten_source * atten_scatter) / scoring_area
    # But we want fluence per unit volume of scatter, integrated over scatter volume
    
    # For integration, we'll compute the contribution per unit volume at scatter point
    # dΦ_scat / dV = ne * ds_dOmega * (atten_source) * (atten_scatter) * (1/r^2) * (projected area factor)
    
    # The fluence at scoring vol from a unit volume at scatter point is:
    dPhi_dV = (ne * ds_dOmega * atten_source_to_scatter * atten_scatter_to_scoring 
               / (r**2)) * abs(dir_to_scoring[2])
    
    return dPhi_dV, E_scat

def compute_scatter_spectrum():
    """Compute first scatter contribution to fluence spectrum via numerical integration"""
    
    scoring_pos = np.array([0.0, 0.0, scoring_center_z])
    
    # Discretize the phantom
    nz = 30
    nr = 15
    nphi = 24
    
    z_edges = np.linspace(0, 15, nz+1)
    r_edges = np.linspace(0, 15, nr+1)
    phi_edges = np.linspace(0, 2*np.pi, nphi+1)
    
    energy_bins = np.linspace(0.1, 2.0, 40)
    
    fluence_spectrum = np.zeros(len(energy_bins)-1)
    
    total_fluence = 0.0
    
    for iz in range(nz):
        z_center = (z_edges[iz] + z_edges[iz+1]) / 2
        dz = z_edges[iz+1] - z_edges[iz]
        
        for ir in range(nr):
            r_center = (r_edges[ir] + r_edges[ir+1]) / 2
            dr = r_edges[ir+1] - r_edges[ir]
            
            # Beam width at this depth
            half_width = field_half * (SSD + z_center) / SSD
            
            # Only consider points within the beam
            if r_center > half_width:
                continue
            
            # For each phi
            for iphi in range(nphi):
                phi_center = (phi_edges[iphi] + phi_edges[iphi+1]) / 2
                dphi = phi_edges[iphi+1] - phi_edges[iphi]
                
                x = r_center * np.cos(phi_center)
                y = r_center * np.sin(phi_center)
                
                if abs(x) > 15 or abs(y) > 15:
                    continue
                
                scatter_pos = np.array([x, y, z_center])
                dV = r_center * dr * dphi * dz
                
                dPhi, E_scat = scatter_to_fluence(E0, scoring_pos, scatter_pos)
                
                if dPhi > 0:
                    contribution = dPhi * dV
                    total_fluence += contribution
                    
                    # Add to energy bin
                    bin_idx = np.digitize(E_scat, energy_bins) - 1
                    if 0 <= bin_idx < len(energy_bins)-1:
                        fluence_spectrum[bin_idx] += contribution
    
    return energy_bins, fluence_spectrum, total_fluence

print("Computing first scatter fluence spectrum...")
# This might be slow, let me reduce resolution
# Actually, let me use a sparser grid first
energy_bins, fluence_spectrum, total_scatter_fluence = compute_scatter_spectrum()

# Primary fluence at scoring volume
primary_fluence = np.exp(-mu_at_E0 * scoring_center_z) * (scoring_area / (2*field_at_depth)**2)
# Actually the primary fluence (relative to Phi0) at the scoring volume is just exp(-μ*d)
# because Phi0 is the fluence at that point without phantom.
# The scoring volume is on the central axis.
print(f"Primary fluence at scoring vol (rel to Phi0): {np.exp(-mu_at_E0 * scoring_center_z):.4f}")
print(f"First scatter fluence at scoring vol (rel to Phi0): {total_scatter_fluence:.4f}")

# Fluence ratio
if total_scatter_fluence > 0:
    print(f"Scatter/Primary fluence ratio: {total_scatter_fluence/np.exp(-mu_at_E0 * scoring_center_z):.4f}")

# Print spectrum
print("\nScatter fluence spectrum (first scatter only):")
for i in range(len(energy_bins)-1):
    if fluence_spectrum[i] > total_scatter_fluence * 0.01:
        print(f"  {energy_bins[i]:.2f}-{energy_bins[i+1]:.2f} MeV: {fluence_spectrum[i]:.4f} ({fluence_spectrum[i]/total_scatter_fluence*100:.1f}%)")