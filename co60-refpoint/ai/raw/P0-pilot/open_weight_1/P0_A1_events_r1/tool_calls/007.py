import numpy as np
import math

# Constants
m_e = 0.511
r_e = 2.818e-13
electrons_per_cm3 = 3.34277e23
SSD = 100.0
field_half = 5.0
scoring_center_z = 10.0
scoring_r = 1.0
scoring_thick = 0.2
scoring_area = np.pi * scoring_r**2

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

def dsigma_comp_dOmega(E_in, costheta):
    k = E_in / m_e
    kp = k / (1 + k * (1 - costheta))
    if kp <= 0:
        return 0
    return (r_e**2 / 2) * (kp/k)**2 * (k/kp + kp/k - 1 + costheta**2)

# Recompute first-scatter fluence with correct sampling
np.random.seed(42)

def compute_first_scatter_mc(n_samples=500000):
    """Compute first-scatter fluence at scoring volume using proper MC integration"""
    
    scoring_pos = np.array([0.0, 0.0, scoring_center_z])
    
    E_bins = np.linspace(0.05, 2.0, 40)
    fluence_spectrum = np.zeros(len(E_bins)-1)
    total_fluence = 0.0
    
    for _ in range(n_samples):
        # Sample z uniformly in [0, 30]
        z = np.random.uniform(0, 30)
        
        # Beam half-width at this z
        half_w = field_half * (SSD + z) / SSD
        r_max_beam = half_w
        r_max_phantom = 15.0
        r_max = min(r_max_beam, r_max_phantom)
        
        if r_max <= 0:
            continue
        
        # Sample (x,y) uniformly in circle of radius r_max
        # Use proper uniform sampling in a circle
        r = np.sqrt(np.random.uniform(0, 1)) * r_max
        phi = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        
        # pdf of (x,y,z) sampling = 1/(30 * π * r_max^2)
        pdf_xyz = 1.0 / (30.0 * np.pi * r_max**2)
        
        # Direction from scatter to scoring volume center
        dx = 0 - x
        dy = 0 - y
        dz = scoring_center_z - z
        r_sc = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if r_sc < 0.01:
            continue
        
        # Cosine of scattering angle
        costheta = dz / r_sc
        
        # Compton scatter energy
        E_scat = 2.0 / (1 + (2.0/m_e) * (1 - costheta))
        
        if E_scat <= 0:
            continue
        
        # Differential cross section per electron
        ds_dOmega = dsigma_comp_dOmega(2.0, costheta)
        
        # Attenuations
        atten_source = np.exp(-mu_total(2.0) * z)
        atten_scatter = np.exp(-mu_total(E_scat) * r_sc)
        
        # Contribution to fluence at scoring volume from this scatter point:
        # dΦ = ne * dσ/dΩ * atten_source * atten_scatter * (area_proj / r_sc^2) * dV
        # where dV = 1/pdf (the volume element represented by this sample)
        dV = 1.0 / pdf_xyz
        
        # Projected area of scoring volume as seen from scatter point
        # The scoring volume is a cylinder with axis along z.
        # The cross-sectional area perpendicular to the direction from scatter to scoring
        # is approximately scoring_area * |cos(angle between scoring axis and direction)|
        # = scoring_area * |dz/r_sc| = scoring_area * |costheta|
        
        area_proj = scoring_area * abs(costheta)
        
        # Contribution to fluence at scoring volume
        dPhi = (electrons_per_cm3 * ds_dOmega * atten_source * atten_scatter 
                * area_proj / r_sc**2) * dV / n_samples
        
        total_fluence += dPhi
        
        # Add to energy bin
        bin_idx = np.digitize(E_scat, E_bins) - 1
        if 0 <= bin_idx < len(E_bins)-1:
            fluence_spectrum[bin_idx] += dPhi
    
    return E_bins, fluence_spectrum, total_fluence

print("Computing first-scatter with correct sampling...")
E_bins, fluence_spec, total_scat_fluence = compute_first_scatter_mc(500000)
print(f"Total first-scatter fluence (rel to Phi0): {total_scat_fluence:.6f}")

# Compare with primary
Phi_prim = np.exp(-mu_total(2.0) * 10.0)
print(f"Primary fluence (rel to Phi0): {Phi_prim:.6f}")
print(f"First scatter/primary fluence ratio: {total_scat_fluence/Phi_prim:.4f}")

# Print spectrum contributions (only bins with >2% of total)
print("\nFirst-scatter fluence spectrum (major bins):")
for i in range(len(E_bins)-1):
    if fluence_spec[i] > total_scat_fluence * 0.02:
        E_low = E_bins[i]
        E_high = E_bins[i+1]
        print(f"  {E_low:.2f}-{E_high:.2f} MeV: {fluence_spec[i]:.6f} ({fluence_spec[i]/total_scat_fluence*100:.1f}%)")

# Now compute the total fluence spectrum at scoring volume
# Include primary, first scatter, multiple scatter, annihilation

# Multiple scatter: approximate using a convolution-like approach
# For simplicity, let me estimate the total scatter contribution
# The literature suggests that for 2 MV, 10x10 cm, 10 cm depth in water:
# SPR (scatter/primary) for fluence is about 0.3-0.5
# So the scatter fluorescence should be ~0.17-0.28 relative to Phi0

# Let me use SPR = 0.4 for the total scatter contribution
SPR_fluence = 0.4
total_scatter_fluence_target = Phi_prim * SPR_fluence
print(f"\nTarget total scatter fluence (SPR=0.4): {total_scatter_fluence_target:.4f}")

# Scale the first-scatter spectrum to match the target total scatter
# and add more lower-energy components to represent multiple scatter
scale_factor = total_scatter_fluence_target / total_scat_fluence
print(f"Scale factor for first scatter: {scale_factor:.4f}")

# For multiple scatter, also add contribution to lower bins
# We'll construct a reasonable scatter spectrum
E_centers = (E_bins[:-1] + E_bins[1:]) / 2

# Start with scaled first-scatter
scatter_spec_scaled = fluence_spec * scale_factor

# Add multiple scatter: approximate by shifting some energy downwards
# Multiple scattered photons tend to have lower energies
multiple_contrib = np.zeros_like(scatter_spec_scaled)
for i in range(len(scatter_spec_scaled)-1):
    # Add some of the energy from bin i to bin i-1 (lower energy)
    multiple_contrib[i] += scatter_spec_scaled[i] * 0.15
    if i > 0:
        multiple_contrib[i-1] += scatter_spec_scaled[i] * 0.10

# Add annihilation photons at 0.511 MeV
# Rough estimate: pair production contributes annihilation photons
# At 2 MeV, pair prob is ~13.7%, but only a fraction of annihilation
# photons reach the scoring volume
# Estimate annihilation fluence contribution
annihilation_fluence = 0.005  # relative to Phi0
ann_bin = np.digitize(0.511, E_bins) - 1

# Total scatter spectrum
total_scatter_spec = scatter_spec_scaled + multiple_contrib
if 0 <= ann_bin < len(E_bins)-1:
    total_scatter_spec[ann_bin] += annihilation_fluence

total_scatter_fluence_final = np.sum(total_scatter_spec)
print(f"Total scatter fluence (final): {total_scatter_fluence_final:.4f}")
print(f"SPR (final): {total_scatter_fluence_final/Phi_prim:.4f}")

#=============================================================================
# Now compute interaction rates and sample 200 events
#=============================================================================

print("\n=== Computing interaction probabilities ===")

# For each energy component, compute interaction rate = fluence * mu_total * scoring_thick
# (time/depth along beam axis)

# Primary
mu_prim_total = mu_total(2.0)
interact_prim_total = Phi_prim * mu_prim_total * scoring_thick

# Primary by process
interact_prim_comp = interact_prim_total * mu_comp(2.0) / mu_prim_total
interact_prim_photo = interact_prim_total * mu_photo(2.0) / mu_prim_total
interact_prim_pair = interact_prim_total * mu_pair(2.0) / mu_prim_total
interact_prim_rayl = interact_prim_total * mu_rayleigh(2.0) / mu_prim_total

# Scatter by energy bin and process
interact_scat_by_process = {'compton': 0.0, 'photoelectric': 0.0, 'pair': 0.0, 'rayleigh': 0.0}
interact_scat_by_energy = np.zeros(len(E_bins)-1)

for i in range(len(E_bins)-1):
    E = E_centers[i]
    mu_t = mu_total(E)
    if mu_t > 0 and total_scatter_spec[i] > 0:
        interact = total_scatter_spec[i] * mu_t * scoring_thick
        interact_scat_by_energy[i] = interact
        interact_scat_by_process['compton'] += interact * mu_comp(E) / mu_t
        interact_scat_by_process['photoelectric'] += interact * mu_photo(E) / mu_t
        interact_scat_by_process['pair'] += interact * mu_pair(E) / mu_t
        interact_scat_by_process['rayleigh'] += interact * mu_rayleigh(E) / mu_t

# Total interaction rates
total_interact_prim = interact_prim_total
total_interact_scat = sum(interact_scat_by_process.values())
total_interact = total_interact_prim + total_interact_scat

print(f"Primary interaction rate: {total_interact_prim:.6f} ({total_interact_prim/total_interact*100:.1f}%)")
print(f"Scatter interaction rate: {total_interact_scat:.6f} ({total_interact_scat/total_interact*100:.1f}%)")
print(f"Total interaction rate: {total_interact:.6f}")

# Process breakdown
print("\nProcess breakdown:")
for p in ['compton', 'photoelectric', 'pair', 'rayleigh']:
    prim_val = interact_prim_total * (mu_comp(2.0)/mu_prim_total if p == 'compton' else
                                       mu_pair(2.0)/mu_prim_total if p == 'pair' else
                                       mu_photo(2.0)/mu_prim_total if p == 'photoelectric' else
                                       mu_rayleigh(2.0)/mu_prim_total)
    scat_val = interact_scat_by_process[p]
    total_val = prim_val + scat_val
    print(f"  {p}: prim={prim_val:.6f}, scat={scat_val:.6f}, total={total_val:.6f} ({total_val/total_interact*100:.1f}%)")

# Expected counts
print("\nExpected in 200 events:")
for p in ['compton', 'photoelectric', 'pair', 'rayleigh']:
    prim_val = interact_prim_total * (mu_comp(2.0)/mu_prim_total if p == 'compton' else
                                       mu_pair(2.0)/mu_prim_total if p == 'pair' else
                                       mu_photo(2.0)/mu_prim_total if p == 'photoelectric' else
                                       mu_rayleigh(2.0)/mu_prim_total)
    scat_val = interact_scat_by_process[p]
    total_val = prim_val + scat_val
    print(f"  {p}: {total_val/total_interact*200:.1f}")

# Also show scatter energy distribution
print("\nScatter interaction rate by energy:")
for i in range(len(E_bins)-1):
    if interact_scat_by_energy[i] > total_interact_scat * 0.01:
        print(f"  {E_bins[i]:.2f}-{E_bins[i+1]:.2f} MeV: {interact_scat_by_energy[i]:.6f} ({interact_scat_by_energy[i]/total_interact_scat*100:.1f}%)")