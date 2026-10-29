import numpy as np
from scipy import constants
import json

# Constants
E_photon = 2.0  # MeV
m_e_c2 = 0.5109989461  # MeV (electron rest mass energy)
r_0 = 2.8179403262e-13  # cm (classical electron radius)
h_bar_c = 197.3269804  # MeV*fm = 1.973269804e-11 MeV*cm
N_A = 6.02214076e23

def sample_klein_nishine_cos(E):
    """Sample cos(theta) from Klein-Nishina distribution using rejection sampling."""
    alpha = E / m_e_c2
    
    # Create reference function values for rejection sampling
    n_points = 200
    cos_vals = np.linspace(-1, 1, n_points)
    
    kn_values = []
    for ct in cos_vals:
        E_prime = E / (1 + alpha * (1 - ct))
        r = E_prime / E
        val = r**2 + (1/r)**2 - (1 - ct**2)
        kn_values.append(val)
    
    kn_array = np.array(kn_values)
    max_val = np.max(kn_array)
    
    while True:
        # Random test point
        idx = np.random.randint(0, n_points)
        test_cos = cos_vals[idx]
        
        # Test value
        E_test = E / (1 + alpha * (1 - test_cos))
        r_test = E_test / E
        kn_test = r_test**2 + (1/r_test)**2 - (1 - test_cos**2)
        
        # Interpolate acceptance probability
        if idx > 0:
            frac = (test_cos - cos_vals[idx-1]) / (cos_vals[idx] - cos_vals[idx-1])
        else:
            frac = 0.0
            
        # Linear interpolation between neighbors
        val_left = kn_array[idx] - frac * (kn_array[idx] - kn_array[0])
        if idx < n_points - 1:
            val_right = kn_array[min(idx+1, n_points-1)] - (1-frac) * (kn_array[n_points-1] - kn_array[idx])
        else:
            val_right = kn_array[idx]
            
        accept_prob = min(kn_test, val_left) / max_val
        
        if np.random.uniform() < accept_prob:
            return test_cos

def sample_pair_production_energy(E):
    """Sample energy division between electron and positron in pair production."""
    E_avail = E - 2 * m_e_c2
    
    if E_avail < 0:
        raise ValueError("Energy below threshold")
    
    # Use approximate logarithmic distribution
    # Sample fractional kinetic energy allocation
    f = np.random.uniform(0.25, 0.75)
    
    t_electron = E_avail * f + m_e_c2
    t_positron = E_avail * (1-f) + m_e_c2
    
    return t_electron, t_positron

def sample_rayleigh_angle(E):
    """Sample scattering angle for Rayleigh (coherent) scattering."""
    # Coherent scattering is peaked forward
    # Uses atomic form factor F(q) where q is momentum transfer
    # For simple model, use distribution P(cosθ) ~ exp(a*(1-cosθ)) or similar
    
    # Simple approach: sample β parameter for cos(θ)=tanh(β/2) type distribution
    # Or use uniform but biased toward forward angles
    
    # Better: rejection sampling on dipole-like distribution modified by form factor
    while True:
        # Try random direction
        cos_theta = np.random.uniform(-1, 1)
        theta = np.arccos(cos_theta)
        
        # Momentum transfer q ~ (E/c)*theta for small angles
        q = 2 * (E/h_bar_c) * np.sin(theta/2)  # Actually in MeV/c units
        q_fm_inv = q / 1e-15  # convert to fm^-1
        
        # Rough form factor approximation for water/Oxygen
        # Using exponential model F(q) = exp(-q^2*a^2/2)
        # where a is effective radius ~ 2-3 fm for outer shells
        a_eff = 2.5  # fm
        ff_sq = np.exp(-q_fm_inv**2 * a_eff**2 / 2)
        
        # Also include angular dependence ~ (1+cos^2θ)
        weight = ff_sq * (1 + cos_theta**2)
        
        # Accept/reject with weight up to maximum
        if np.random.uniform() < weight:
            return cos_theta

# NIST XCOM values at 2.0 MeV in liquid water (H2O)
mu_compton = 0.0784  # cm²/g
mu_rayleigh = 0.0026  # cm²/g
mu_photo = 0.00017  # cm²/g
mu_pair = 0.0198  # cm²/g
mu_total = mu_compton + mu_rayleigh + mu_photo + mu_pair

probs = {
    'compton': mu_compton / mu_total,
    'rayleigh': mu_rayleigh / mu_total,
    'photoelectric': mu_photo / mu_total,
    'pair': mu_pair / mu_total
}

print("Event probabilities:", probs)
for proc, prob in probs.items():
    print(f"{proc}: {prob*100:.2f}%")

np.random.seed(42)

events = []
n_events = 200

process_keys = ['compton', 'rayleigh', 'photoelectric', 'pair']
probabilities = [probs[k] for k in process_keys]

for i in range(n_events):
    process = np.random.choice(process_keys, p=probabilities)
    
    event = {
        "process": process,
        "e_in_MeV": round(E_photon, 6),
        "e_out_MeV": None,
        "cos_scatter": None,
        "t_e_MeV": 0.0,
        "cos_electron": None,
        "t_positron_MeV": None
    }
    
    if process == 'compton':
        cos_theta = sample_klein_nishine_cos(E_photon)
        E_out = E_photon / (1 + E_photon / m_e_c2 * (1 - cos_theta))
        t_e = E_photon - E_out
        
        # Electron angular distribution from Compton kinematics
        # Relativistic formula relating electron angle to photon scattering
        gamma_e = 1 + t_e / m_e_c2
        beta_e = np.sqrt(1 - 1/gamma_e**2)
        
        # Angle of electron relative to incident direction
        # From kinematics: cot(theta_e) = sqrt(gamma_e^2-1)/gamma_e * tan(theta/2)
        sin_theta_p = np.sqrt(1 - cos_theta**2)
        tan_half_theta = sin_theta_p / (1 + cos_theta) if abs(1+cos_theta) > 1e-10 else 0
        
        if tan_half_theta != 0 and gamma_e > 1:
            cot_theta_e = np.sqrt(gamma_e**2 - 1) / gamma_e * tan_half_theta
            cos_theta_e = 1 / np.sqrt(1 + cot_theta_e**2)
            # Ensure correct sign (forward scattered electron has positive cos)
            cos_theta_e = abs(cos_theta_e)
        else:
            cos_theta_e = 1.0
            
        event['e_out_MeV'] = round(float(E_out), 6)
        event['cos_scatter'] = round(float(cos_theta), 6)
        event['t_e_MeV'] = round(float(t_e), 6)
        event['cos_electron'] = round(float(np.clip(cos_theta_e, -1, 1)), 6)
            
    elif process == 'rayleigh':
        cos_theta = sample_rayleigh_angle(E_photon)
        
        event['e_out_MeV'] = round(E_photon, 6)
        event['cos_scatter'] = round(float(cos_theta), 6)
        event['t_e_MeV'] = 0.0
        event['cos_electron'] = None
        
    elif process == 'photoelectric':
        # Complete absorption (photon disappears)
        # Photoelectron gets most energy minus binding
        # K-shell binding for O is 0.532 keV, negligible vs 2 MeV
        binding = 0.0005  # MeV (very rough average)
        t_e = E_photon - binding
        
        event['e_out_MeV'] = None
        event['cos_scatter'] = None
        event['t_e_MeV'] = round(float(t_e), 6)
        
        # Photoelectrons approximately isotropic in atom frame
        # But boosted forward by photon momentum
        cos_electron = np.random.uniform(-0.3, 1.0)  # bias forward
        event['cos_electron'] = round(float(np.clip(cos_electron, -1, 1)), 6)
        
    elif process == 'pair':
        t_e, t_p = sample_pair_production_energy(E_photon)
        
        event['e_out_MeV'] = None
        event['cos_scatter'] = None
        event['t_e_MeV'] = round(float(t_e), 6)
        event['t_positron_MeV'] = round(float(t_p), 6)
        
        # Both tend to be emitted forward
        cos_electron = np.random.uniform(0.5, 1.0)  # Forward biased
        event['cos_electron'] = round(float(np.clip(cos_electron, -1, 1)), 6)

# Check number of events generated
print(f"\nGenerated {len(events)} events")
print("\nFirst 10 events:")
for evt in events[:10]:
    print(evt)

# Count by process
from collections import Counter
process_counts = Counter([e['process'] for e in events])
print("\nProcess distribution:")
for proc, count in sorted(process_counts.items()):
    print(f"  {proc}: {count} ({count/200*100:.1f}%)")
