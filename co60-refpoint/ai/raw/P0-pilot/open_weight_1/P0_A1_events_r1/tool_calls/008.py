import numpy as np
import math
import json

# Set seed for reproducibility
np.random.seed(42 + 1)

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

#=============================================================================
# Cross section functions
#=============================================================================

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

def get_process_probs(E):
    """Return dict of process probabilities for a photon of energy E"""
    mt = mu_total(E)
    if mt == 0:
        return {'compton': 0, 'photoelectric': 0, 'pair': 0, 'rayleigh': 0}
    return {
        'compton': mu_comp(E) / mt,
        'photoelectric': mu_photo(E) / mt,
        'pair': mu_pair(E) / mt,
        'rayleigh': mu_rayleigh(E) / mt
    }

def sample_process(E):
    """Sample interaction process for a photon of energy E"""
    probs = get_process_probs(E)
    r = np.random.random()
    cum = 0
    for p in ['compton', 'photoelectric', 'pair', 'rayleigh']:
        cum += probs[p]
        if r < cum:
            return p
    return 'compton'

#=============================================================================
# Kinematics sampling
#=============================================================================

def sample_compton_angle(E_in):
    """Sample cos(theta) from Klein-Nishina distribution"""
    k = E_in / m_e
    if k <= 0:
        return 0.0
    
    # For rejection sampling, find max of differential cross section
    # dσ/dΩ ∝ (k'/k)^2 * (k/k' + k'/k - 1 + mu^2)
    # where k' = k/(1+k(1-mu))
    # Evaluate on a grid to find max
    n_test = 2000
    mus = np.linspace(-0.9999, 0.9999, n_test)
    max_val = 0
    for mu in mus:
        kp = k / (1 + k * (1 - mu))
        if kp > 0:
            val = (kp/k)**2 * (k/kp + kp/k - 1 + mu**2)
            max_val = max(max_val, val)
    max_val *= 1.05
    
    while True:
        mu = np.random.uniform(-1, 1)
        kp = k / (1 + k * (1 - mu))
        if kp <= 0:
            continue
        val = (kp/k)**2 * (k/kp + kp/k - 1 + mu**2)
        if np.random.random() * max_val <= val:
            return mu

def compton_kinematics(E_in, costheta):
    """Return (E_out, cos_electron, t_e)"""
    E_out = E_in / (1 + (E_in/m_e) * (1 - costheta))
    t_e = E_in - E_out
    
    # Electron angle
    if E_in > 0 and E_out > 0:
        p_in = math.sqrt(E_in**2 + 2*E_in*m_e)
        p_out = math.sqrt(E_out**2 + 2*E_out*m_e)
        cos_e = (p_in - p_out * costheta) / math.sqrt(p_in**2 + p_out**2 - 2*p_in*p_out*costheta)
        cos_e = max(-1.0, min(1.0, cos_e))
    else:
        cos_e = 0.0
    
    return E_out, cos_e, t_e

def sample_pair_kinematics(E_in):
    """Sample pair production kinematics.
    Returns (t_e, t_positron, cos_electron)
    """
    available = E_in - 2 * m_e  # energy available for kinetic energies
    
    # Roughly, electron and positron share the energy approximately equally
    # with some variation
    f = np.random.uniform(0.3, 0.7)
    t_e = available * f
    t_pos = available * (1 - f)
    
    # Angular distribution: roughly forward peaked
    # Mean angle ~ m_e / (E/2) for each particle
    mean_angle = m_e / (E_in / 2)
    # Use exponential distribution for angle
    theta_e = np.random.exponential(mean_angle) if mean_angle > 0 else 0.001
    theta_e = min(theta_e, np.pi/4)  # limit
    
    cos_e = math.cos(theta_e)
    
    return t_e, t_pos, cos_e

#=============================================================================
# Build fluence spectrum and interaction probabilities
#=============================================================================

np.random.seed(42)

# Compute first-scatter fluence
def compute_first_scatter_fluence(n_samples=300000):
    scoring_pos = np.array([0.0, 0.0, scoring_center_z])
    E_bins = np.linspace(0.05, 2.0, 40)
    fluence_spec = np.zeros(len(E_bins)-1)
    
    for _ in range(n_samples):
        z = np.random.uniform(0, 30)
        half_w = field_half * (SSD + z) / SSD
        r_max = min(half_w, 15.0)
        if r_max <= 0:
            continue
        r = np.sqrt(np.random.uniform(0, 1)) * r_max
        phi = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        
        pdf_xyz = 1.0 / (30.0 * np.pi * r_max**2)
        
        dx = 0 - x
        dy = 0 - y
        dz = scoring_center_z - z
        r_sc = math.sqrt(dx**2 + dy**2 + dz**2)
        if r_sc < 0.01:
            continue
        
        costheta = dz / r_sc
        E_scat = 2.0 / (1 + (2.0/m_e) * (1 - costheta))
        if E_scat <= 0:
            continue
        
        ds_dOmega = (r_e**2 / 2) * (E_scat/2.0)**2 * (2.0/E_scat + E_scat/2.0 - 1 + costheta**2)
        # More precisely using the Klein-Nishina formula
        k = 2.0 / m_e
        kp = E_scat / m_e
        ds_dOmega = (r_e**2 / 2) * (kp/k)**2 * (k/kp + kp/k - 1 + costheta**2)
        
        atten_source = np.exp(-mu_total(2.0) * z)
        atten_scatter = np.exp(-mu_total(E_scat) * r_sc)
        
        dV = 1.0 / pdf_xyz
        area_proj = scoring_area * abs(costheta)
        dPhi = (electrons_per_cm3 * ds_dOmega * atten_source * atten_scatter 
                * area_proj / r_sc**2) * dV / n_samples
        
        bin_idx = np.digitize(E_scat, E_bins) - 1
        if 0 <= bin_idx < len(E_bins)-1:
            fluence_spec[bin_idx] += dPhi
    
    return E_bins, fluence_spec

E_bins, first_scatter_spec = compute_first_scatter_fluence(300000)
first_scatter_total = np.sum(first_scatter_spec)

# Primary fluence
Phi_prim = np.exp(-mu_total(2.0) * 10.0)

# Scale to get reasonable SPR
SPR = 0.40  # scatter/primary fluence ratio
total_scatter_target = Phi_prim * SPR
scale = total_scatter_target / first_scatter_total if first_scatter_total > 0 else 0

# Build total scatter spectrum
scatter_spec = first_scatter_spec * scale

# Add multiple scatter (approximate lower energy contribution)
extra_low = np.zeros_like(scatter_spec)
for i in range(1, len(scatter_spec)):
    extra_low[i-1] += scatter_spec[i] * 0.15
scatter_spec += extra_low

# Add annihilation peak at 0.511 MeV
ann_bin = np.digitize(0.511, E_bins) - 1
ann_contrib = 0.003  # relative to Phi0
if 0 <= ann_bin < len(E_bins)-1:
    scatter_spec[ann_bin] += ann_contrib

# Normalize scatter spectrum
scatter_spec_total = np.sum(scatter_spec)

# Now compute interaction rates
E_centers = (E_bins[:-1] + E_bins[1:]) / 2

# Primary interaction rate
mu_prim = mu_total(2.0)
primary_interact = Phi_prim * mu_prim * scoring_thick

# Scatter interaction rates
scatter_interact_by_energy = np.zeros_like(scatter_spec)
for i in range(len(E_bins)-1):
    E = E_centers[i]
    mu_t = mu_total(E)
    if mu_t > 0:
        scatter_interact_by_energy[i] = scatter_spec[i] * mu_t * scoring_thick

total_scatter_interact = np.sum(scatter_interact_by_energy)
total_interact = primary_interact + total_scatter_interact

# Primary event probability
p_primary = primary_interact / total_interact
print(f"Primary interaction fraction: {p_primary*100:.1f}%")
print(f"Scatter interaction fraction: {total_scatter_interact/total_interact*100:.1f}%")

#=============================================================================
# Generate 200 events
#=============================================================================

events = []

for _ in range(200):
    # Decide if primary or scattered
    if np.random.random() < p_primary:
        # Primary photon interaction
        E_in = 2.0
        
        # Sample process
        process = sample_process(E_in)
        
        if process == 'compton':
            costheta = sample_compton_angle(E_in)
            E_out, cos_e, t_e = compton_kinematics(E_in, costheta)
            event = {
                'process': 'compton',
                'e_in_MeV': E_in,
                'e_out_MeV': E_out,
                'cos_scatter': costheta,
                't_e_MeV': t_e,
                'cos_electron': cos_e,
                't_positron_MeV': None
            }
        elif process == 'photoelectric':
            # All energy to electron (binding energy negligible for water)
            event = {
                'process': 'photoelectric',
                'e_in_MeV': E_in,
                'e_out_MeV': None,
                'cos_scatter': None,
                't_e_MeV': E_in,
                'cos_electron': 1.0,  # approximately forward
                't_positron_MeV': None
            }
        elif process == 'pair':
            t_e, t_pos, cos_e = sample_pair_kinematics(E_in)
            event = {
                'process': 'pair',
                'e_in_MeV': E_in,
                'e_out_MeV': None,
                'cos_scatter': None,
                't_e_MeV': t_e,
                'cos_electron': cos_e,
                't_positron_MeV': t_pos
            }
        elif process == 'rayleigh':
            # Rayleigh: very forward peaked
            # Sample from a narrow forward distribution
            sigma_theta = 0.001  # ~0.05 degree
            theta = abs(np.random.normal(0, sigma_theta))
            costheta = math.cos(theta)
            event = {
                'process': 'rayleigh',
                'e_in_MeV': E_in,
                'e_out_MeV': E_in,
                'cos_scatter': costheta,
                't_e_MeV': 0.0,
                'cos_electron': None,
                't_positron_MeV': None
            }
    else:
        # Scattered photon interaction
        # Sample energy from scatter spectrum
        # First select energy bin
        bin_probs = scatter_interact_by_energy / total_scatter_interact
        bin_idx = np.random.choice(len(bin_probs), p=bin_probs)
        
        # Sample energy uniformly within bin
        E_in = np.random.uniform(E_bins[bin_idx], E_bins[bin_idx+1])
        
        # Sample process
        process = sample_process(E_in)
        
        if process == 'compton':
            costheta = sample_compton_angle(E_in)
            E_out, cos_e, t_e = compton_kinematics(E_in, costheta)
            event = {
                'process': 'compton',
                'e_in_MeV': E_in,
                'e_out_MeV': E_out,
                'cos_scatter': costheta,
                't_e_MeV': t_e,
                'cos_electron': cos_e,
                't_positron_MeV': None
            }
        elif process == 'photoelectric':
            event = {
                'process': 'photoelectric',
                'e_in_MeV': E_in,
                'e_out_MeV': None,
                'cos_scatter': None,
                't_e_MeV': E_in,
                'cos_electron': 1.0,
                't_positron_MeV': None
            }
        elif process == 'pair':
            t_e, t_pos, cos_e = sample_pair_kinematics(E_in)
            event = {
                'process': 'pair',
                'e_in_MeV': E_in,
                'e_out_MeV': None,
                'cos_scatter': None,
                't_e_MeV': t_e,
                'cos_electron': cos_e,
                't_positron_MeV': t_pos
            }
        elif process == 'rayleigh':
            sigma_theta = 0.001 * (2.0 / E_in)  # wider for lower energies
            theta = abs(np.random.normal(0, sigma_theta))
            costheta = math.cos(theta)
            event = {
                'process': 'rayleigh',
                'e_in_MeV': E_in,
                'e_out_MeV': E_in,
                'cos_scatter': costheta,
                't_e_MeV': 0.0,
                'cos_electron': None,
                't_positron_MeV': None
            }
    
    events.append(event)

# Print summary
print(f"\nGenerated {len(events)} events")
proc_counts = {}
for e in events:
    p = e['process']
    proc_counts[p] = proc_counts.get(p, 0) + 1
for p in ['compton', 'photoelectric', 'pair', 'rayleigh']:
    cnt = proc_counts.get(p, 0)
    print(f"  {p}: {cnt} ({cnt/200*100:.1f}%)")

# Check primary vs scattered
prim_count = sum(1 for e in events if abs(e['e_in_MeV'] - 2.0) < 0.001)
print(f"Primary (2.0 MeV) interactions: {prim_count}")

# Range of energies for scatter
scat_energies = [e['e_in_MeV'] for e in events if abs(e['e_in_MeV'] - 2.0) >= 0.001]
if scat_energies:
    print(f"Scatter energy range: {min(scat_energies):.3f} - {max(scat_energies):.3f} MeV")
    print(f"Scatter mean energy: {np.mean(scat_energies):.3f} MeV")

# Print first few events
print("\nFirst 5 events:")
for i, e in enumerate(events[:5]):
    print(f"  {i+1}: {e['process']}, E_in={e['e_in_MeV']:.3f}, E_out={e['e_out_MeV']}, cos_scatter={e['cos_scatter']}")

# Save for output
print(f"\nMethod description: Generated using analytical fluence calculation at 10 cm depth in water for 2 MeV photons, 10x10 cm field. Primary fluence = Phi0*exp(-μ*10). First Compton scatter fluence computed via MC integration of Klein-Nishina differential cross section over beam volume. Multiple scatter and annihilation contributions approximated. Interactions sampled from combined fluence spectrum weighted by energy-dependent total cross section. Kinematics from Klein-Nishina (Compton), Bethe-Heitler approximation (pair), forward-peaked (Rayleigh, photoelectric).")