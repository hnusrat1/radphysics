import numpy as np
import math

# Constants
m_e = 0.511
r_e = 2.818e-13
electrons_per_cm3 = 3.34277e23

# Cross sections (same as before)
def mu_comp(E):
    if E <= 0:
        return 0
    epsilon = E / m_e
    k = epsilon
    sigma = 2 * np.pi * r_e**2 * (
        (1 + k) / k**2 * (2 * (1 + k) / (1 + 2*k) - np.log(1 + 2*k) / k) +
        np.log(1 + 2*k) / (2*k) - (1 + 3*k) / (1 + 2*k)**2
    )
    return sigma * electrons_per_cm3

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
# First-scatter fluence calculation (more refined)
#=============================================================================

def dsigma_comp_dOmega(E_in, costheta):
    """Klein-Nishina differential cross section per electron (cm^2/sr)"""
    k = E_in / m_e
    kp = k / (1 + k * (1 - costheta))
    if kp <= 0:
        return 0
    return (r_e**2 / 2) * (kp/k)**2 * (k/kp + kp/k - 1 + costheta**2)

# Compute first-scatter fluence more efficiently
# We'll sample points and compute contributions

np.random.seed(42)

# First, compute primary and first-scatter fluence 
scoring_center_z = 10.0
scoring_r = 1.0
scoring_thick = 0.2

# Primary fluence (relative to Phi0) at scoring volume
# On central axis, at depth 10 cm
Phi_prim = np.exp(-mu_total(2.0) * scoring_center_z)
print(f"Primary fluence (rel to Phi0): {Phi_prim:.4f}")

# Now compute first-scatter fluence using MC integration
# with more samples
def compute_first_scatter_spectrum_mc(n_samples=500000):
    """Compute first scatter fluence spectrum at scoring volume using MC"""
    
    scoring_pos = np.array([0.0, 0.0, scoring_center_z])
    
    # Energy bins
    E_bins = np.linspace(0.05, 2.0, 40)
    E_centers = (E_bins[:-1] + E_bins[1:]) / 2
    
    fluence_spectrum = np.zeros(len(E_bins)-1)
    total_fluence = 0.0
    
    # For MC integration, sample scatter points within the phantom beam volume
    # The weight of each sample is: dσ/dΩ * ne * exp(-μ_source*d1) * exp(-μ_scat*d2) / r^2 * area_factor / pdf_of_sampling
    
    for _ in range(n_samples):
        # Sample scatter point within the beam + phantom
        # z from 0 to 30
        z = np.random.uniform(0, 30)
        
        # Beam width at this z
        half_w = field_half * (SSD + z) / SSD
        r_max = min(half_w, 15.0)
        
        # Sample x,y uniformly within beam cross-section
        r = np.random.uniform(0, r_max)
        phi = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        
        # Check if in phantom
        if abs(x) > 15 or abs(y) > 15:
            continue
        
        # PDF of sampling: 1/(30 * π * r_max^2)
        pdf = 1.0 / (30 * np.pi * r_max**2)
        
        # Direction from scatter point to scoring volume center
        dx = 0 - x
        dy = 0 - y
        dz = scoring_center_z - z
        r_sc = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if r_sc < 0.01:
            continue
        
        # Cosine of scattering angle (angle between incident direction (+z) and scattered direction to scoring)
        costheta = dz / r_sc
        
        # Energy after Compton scatter
        E_scat = 2.0 / (1 + (2.0/m_e) * (1 - costheta))
        
        if E_scat <= 0:
            continue
        
        # Differential cross section
        ds_dOmega = dsigma_comp_dOmega(2.0, costheta)
        
        # Attenuation from source to scatter point
        mu_prim = mu_total(2.0)
        atten_source = np.exp(-mu_prim * z)
        
        # Attenuation from scatter point to scoring volume
        mu_scat = mu_total(E_scat)
        atten_scatter = np.exp(-mu_scat * r_sc)
        
        # Contribution to fluence at scoring volume (per unit volume at scatter point)
        # dΦ = ne * ds_dΩ * atten_source * atten_scatter * (scoring_area_proj / r^2) * dV
        # scoring_area_proj = scoring_area * |cosine between direction to scoring and scoring axis|
        # The scoring volume axis is z, and the direction to scoring has z-component dz/r_sc
        # So the projected area = scoring_area * |dz/r_sc| = scoring_area * |costheta|
        
        dV_weight = 1.0 / pdf  # 1/pdf per sample
        
        dPhi = (electrons_per_cm3 * ds_dOmega * atten_source * atten_scatter 
                * scoring_area * abs(costheta) / r_sc**2) * dV_weight / n_samples
        
        total_fluence += dPhi
        
        # Add to energy bin
        bin_idx = np.digitize(E_scat, E_bins) - 1
        if 0 <= bin_idx < len(E_bins)-1:
            fluence_spectrum[bin_idx] += dPhi
    
    return E_bins, fluence_spectrum, total_fluence

SSD = 100.0
field_half = 5.0
scoring_area = np.pi * scoring_r**2

print("\nComputing first-scatter fluence spectrum via MC...")
E_bins, fluence_spec, total_scat_fluence = compute_first_scatter_spectrum_mc(200000)
print(f"Total first-scatter fluence (rel to Phi0): {total_scat_fluence:.4f}")

# Print spectrum
print("\nFirst-scatter fluence spectrum:")
for i in range(len(E_bins)-1):
    if fluence_spec[i] > total_scat_fluence * 0.01:
        print(f"  {E_bins[i]:.2f}-{E_bins[i+1]:.2f} MeV: {fluence_spec[i]:.4f} ({fluence_spec[i]/total_scat_fluence*100:.1f}%)")

# Now compute interaction rates
# For each energy E, the interaction rate in scoring vol is:
# Rate(E) = fluence(E) * mu_total(E) * scoring_thick
# (scoring_thick is the thickness along beam axis, 
#  but for scattered photons coming from different directions, 
#  the effective path length varies)
# For simplicity, approximate with scoring_thick for all

print("\n=== Interaction rates in scoring volume ===")

# Primary interactions
mu_prim = mu_total(2.0)
primary_fluence = Phi_prim
primary_interact_rate = primary_fluence * mu_prim * scoring_thick
print(f"Primary interaction rate: {primary_interact_rate:.6f}")

# First-scatter interactions
scatter_interact_rate = 0.0
for i in range(len(E_bins)-1):
    E = (E_bins[i] + E_bins[i+1]) / 2
    mu = mu_total(E)
    interact_rate_i = fluence_spec[i] * mu * scoring_thick
    scatter_interact_rate += interact_rate_i

print(f"First-scatter interaction rate: {scatter_interact_rate:.6f}")

# For multiple scatters, let me approximate
# Typically multiple scatter adds ~20-50% to single scatter
# Let me estimate by assuming the multiple scatter contribution
# follows a similar energy spectrum but shifted to lower energies

# Multiple scatter approximation
# The ratio of multiple to single scatter depends on beam energy, field size, depth
# For 2 MeV, 10x10, 10 cm depth, typical SPR (total scatter/primary) for energy fluence
# is around 0.3-0.5 for dose. For fluence it could be higher.
# 
# Actually, let me use a simple model where multiple scatter adds
# approximately 40% of the single-scatter contribution

multiple_scale = 0.4  # rough estimate of multiple/single ratio
print(f"Multiple scatter estimate (40% of single): {scatter_interact_rate * multiple_scale:.6f}")

# Also add annihilation photons from pair production
# Pair production at 2 MeV gives one electron and one positron
# Positron annihilates producing two 0.511 MeV photons
# The annihilation photons contribute to the photon fluence
# This is small but non-zero

# For pair production, the cross section is mu_pair(2.0) = 0.0078 cm^-1
# The number of pair productions in the phantom creates annihilation photons
# that can reach the scoring volume
# Very roughly, estimate the annihilation contribution

# Let me construct the total fluence spectrum
# I'll use:
# - Primary: delta function at 2.0 MeV (in the highest energy bin)
# - First scatter: from MC calculation
# - Multiple scatter: approximate by smoothing first scatter to lower energies
# - Annihilation: peak at 0.511 MeV

# Multiple scatter approximation: convolve first scatter spectrum
# For simplicity, I'll just add a fraction of first scatter to lower bins

multiple_spectrum = np.zeros_like(fluence_spec)
for i in range(len(E_bins)-2):
    # Shift one bin down in energy
    if i > 0:
        multiple_spectrum[i-1] += fluence_spec[i] * 0.3
    multiple_spectrum[i] += fluence_spec[i] * 0.1

# Add multiple scatter contribution
total_scatter_spec = fluence_spec + multiple_spectrum

# Add annihilation peak at 0.511 MeV
annihilation_fluence = 0.005  # rough estimate, rel to Phi0
ann_bin = np.digitize(0.511, E_bins) - 1
if 0 <= ann_bin < len(E_bins)-1:
    total_scatter_spec[ann_bin] += annihilation_fluence

total_scatter_fluence_est = np.sum(total_scatter_spec)
print(f"\nTotal scatter fluence estimate: {total_scatter_fluence_est:.4f}")
print(f"Scatter/Primary fluence ratio: {total_scatter_fluence_est/Phi_prim:.4f}")

# Compute total interaction rates
primary_interactions = Phi_prim * mu_total(2.0) * scoring_thick

scatter_interactions = 0.0
scatter_by_energy = np.zeros(len(E_bins)-1)
for i in range(len(E_bins)-1):
    E = (E_bins[i] + E_bins[i+1]) / 2
    mu = mu_total(E)
    scatter_interactions += total_scatter_spec[i] * mu * scoring_thick
    scatter_by_energy[i] = total_scatter_spec[i] * mu * scoring_thick

total_interactions = primary_interactions + scatter_interactions
print(f"\nPrimary interactions (relative units): {primary_interactions:.6f}")
print(f"Scatter interactions (relative units): {scatter_interactions:.6f}")
print(f"Total interactions (relative units): {total_interactions:.6f}")
print(f"Primary fraction: {primary_interactions/total_interactions*100:.1f}%")
print(f"Scatter fraction: {scatter_interactions/total_interactions*100:.1f}%")

# Now compute process breakdown for each energy component
print("\n=== Process breakdown ===")

# For primary (2.0 MeV):
mu_p = mu_total(2.0)
print(f"Primary (2.0 MeV):")
print(f"  Compton: {mu_comp(2.0)/mu_p*100:.1f}%")
print(f"  Pair: {mu_pair(2.0)/mu_p*100:.1f}%")
print(f"  Photo: {mu_photo(2.0)/mu_p*100:.1f}%")
print(f"  Rayleigh: {mu_rayleigh(2.0)/mu_p*100:.1f}%")

# For scatter (average across spectrum)
scatter_interact_by_process = {'compton': 0.0, 'photoelectric': 0.0, 'pair': 0.0, 'rayleigh': 0.0}
for i in range(len(E_bins)-1):
    if total_scatter_spec[i] > 0:
        E = (E_bins[i] + E_bins[i+1]) / 2
        mu_t = mu_total(E)
        if mu_t > 0:
            interact = total_scatter_spec[i] * mu_t * scoring_thick
            scatter_interact_by_process['compton'] += interact * mu_comp(E)/mu_t
            scatter_interact_by_process['photoelectric'] += interact * mu_photo(E)/mu_t
            scatter_interact_by_process['pair'] += interact * mu_pair(E)/mu_t
            scatter_interact_by_process['rayleigh'] += interact * mu_rayleigh(E)/mu_t

print(f"\nScatter interactions (all energies):")
total_scat_proc = sum(scatter_interact_by_process.values())
for p in ['compton', 'photoelectric', 'pair', 'rayleigh']:
    print(f"  {p}: {scatter_interact_by_process[p]:.6f} ({scatter_interact_by_process[p]/total_scat_proc*100:.1f}%)")

# Combined
print(f"\nCombined interactions:")
total_comb = primary_interactions + total_scat_proc
for p in ['compton', 'photoelectric', 'pair', 'rayleigh']:
    prim_p = primary_interactions * (mu_comp(2.0)/mu_p if p == 'compton' else 
                                      mu_pair(2.0)/mu_p if p == 'pair' else
                                      mu_photo(2.0)/mu_p if p == 'photoelectric' else
                                      mu_rayleigh(2.0)/mu_p)
    comb = prim_p + scatter_interact_by_process[p]
    print(f"  {p}: {comb:.6f} ({comb/total_comb*100:.1f}%)")

print(f"  Total: {total_comb:.6f}")
print(f"  Expected counts in 200 events:")
for p in ['compton', 'photoelectric', 'pair', 'rayleigh']:
    prim_p = primary_interactions * (mu_comp(2.0)/mu_p if p == 'compton' else 
                                      mu_pair(2.0)/mu_p if p == 'pair' else
                                      mu_photo(2.0)/mu_p if p == 'photoelectric' else
                                      mu_rayleigh(2.0)/mu_p)
    comb = prim_p + scatter_interact_by_process[p]
    print(f"  {p}: {comb/total_comb*200:.1f}")