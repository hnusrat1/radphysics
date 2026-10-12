import numpy as np
from scipy import constants
import json

# Constants
E_photon = 2.0  # MeV
m_e_c2 = 0.5109989461  # MeV (electron rest mass energy)
r_0 = 2.8179403262e-13  # cm (classical electron radius)
N_A = 6.02214076e23

def sample_klein_nishine_cos(E):
    """Sample cos(theta) from Klein-Nishina distribution."""
    alpha = E / m_e_c2
    
    # Rejection sampling from Klein-Nishina distribution
    # dσ/dΩ ∝ (E'/E)^2 + (E/E')^2 - sin^2(θ)
    # where E' = E/(1+α(1-cosθ))
    
    max_accept = 0
    n_trials = 100000
    
    # Find maximum of acceptance function
    cos_vals = np.linspace(-1, 1, 1000)
    kn_values = []
    for ct in cos_vals:
        E_prime = E / (1 + alpha * (1 - ct))
        r = E_prime / E
        val = r**2 + (1/r)**2 - (1 - ct**2)
        kn_values.append(val)
    
    max_val = max(kn_values)
    
    while True:
        # Random test point
        test_cos = np.random.uniform(-1, 1)
        
        # Test point value
        E_test = E / (1 + alpha * (1 - test_cos))
        r_test = E_test / E
        kn_test = r_test**2 + (1/r_test)**2 - (1 - test_cos**2)
        
        # Acceptance probability
        accept_prob = kn_test / max_val
        
        if np.random.uniform() < accept_prob:
            return test_cos

def sample_pair_production_energy(E):
    """Sample energy division between electron and positron in pair production."""
    # Available energy after creating the pair
    E_avail = E - 2 * m_e_c2
    
    # Rough approximation: symmetric-ish distribution
    # More precisely, it's complex but roughly logarithmic near threshold
    # For a first approximation, sample asymmetrically
    
    if E_avail < 0:
        raise ValueError("Energy below pair production threshold")
    
    # Sample the fraction of available energy going to electron
    # Using a simple model that favors slight asymmetry
    f_el = np.random.uniform(0.5, 1.0)
    
    E_kin_electron = E_avail * (f_el - 0.5) + m_e_c2  # Add rest mass
    E_kin_positron = E_avail * (1 - f_el + 0.5) + m_e_c2
    
    return E_kin_electron, E_kin_positron

def sample_rayleigh_angle(E):
    """Sample scattering angle for Rayleigh (coherent) scattering."""
    # Rayleigh scattering has a peaked forward distribution
    # Uses atomic form factors. Simple approximation using dipole-like distribution
    # P(cosθ) ∝ 1 + cos^2(θ) approximately, but more peaked at low angles
    
    # For simplicity, use a distribution that peaks forward
    # Sample β such that cos(θ) = tanh(β) with appropriate β
    beta_param = 2.0  # Sharpness parameter
    
    # Alternative: rejection sampling on simplified form factor
    cos_vals = np.linspace(0, 1, 100)
    form_factor_sq = []
    for ct in cos_vals:
        # Simplified form factor behavior
        q = 2 * E / h_bar_c * np.sin(np.arccos(ct)/2)  # momentum transfer
        # Very rough form factor F(q) ∝ exp(-q^2/2)
        qq = q * 0.5  # scaled
        ff = np.exp(-qq**2)
        form_factor_sq.append(ff * (1 + ct**2))
    
    form_factor_sq = np.array(form_factor_sq)
    
    while True:
        test_idx = np.random.randint(0, len(cos_vals)-1)
        test_cos = cos_vals[test_idx]
        test_val = form_factor_sq[test_idx]
        
        if np.random.uniform(0, max(form_factor_sq)) < test_val:
            return test_cos

# Get NIST XCOM values for better precision
# From NIST XCOM at 2.0 MeV in H2O:
# Total mass attenuation coefficient: 0.0833 cm²/g
# Coherent (Rayleigh): 0.0026 cm²/g
# Incoherent (Compton): 0.0784 cm²/g
# Photoelectric: 0.00017 cm²/g  
# Pair prod. (total): 0.0198 cm²/g (of which nuclear field is dominant)

mu_compton = 0.0784
mu_rayleigh = 0.0026
mu_photo = 0.00017
mu_pair = 0.0198
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
        
        # Direction of scattered electron
        # Approximate: mostly forward, but depends on scattering angle
        # Use energy-momentum conservation
        if abs(t_e) > 1e-10:
            # cos_electron related to scattering kinematics
            # For high energies, electron goes mainly forward
            # Approximate relation from kinematics
            gamma = 1 + t_e / m_e_c2
            cos_electron_approx = (gamma + 1) / (2 * gamma) if gamma > 1 else 0.5
            
            # Better approximation using relativistic formula
            cos_theta_e = ((1 + 2*(E_photon/m_e_c2)**2*(1-cos_theta)/(E_photon/m_e_c2*(1-cos_theta)+1))**-1)**0.5
            # Actually let's use simpler empirical relation
            cos_electron_approx = min(1.0, max(-1.0, 0.5 + 0.5 * t_e / E_photon))
            
            event['e_out_MeV'] = round(E_out, 6)
            event['cos_scatter'] = round(float(cos_theta), 6)
            event['t_e_MeV'] = round(t_e, 6)
            event['cos_electron'] = round(float(cos_electron_approx), 6)
        else:
            event['cos_electron'] = 1.0
            event['t_e_MeV'] = 0.0
            
    elif process == 'rayleigh':
        # Coherent scattering - photon energy unchanged
        cos_theta = sample_rayleigh_angle(E_photon)
        
        event['e_out_MeV'] = round(E_photon, 6)
        event['cos_scatter'] = round(float(cos_theta), 6)
        event['t_e_MeV'] = 0.0  # No electron produced
        event['cos_electron'] = None
        
    elif process == 'photoelectric':
        # Complete absorption
        # Binding energy is small (~0.5 keV for O K-shell vs 2 MeV)
        binding_energy = 0.000533  # O K-shell in eV -> MeV (very rough)
        t_e = E_photon - binding_energy
        
        event['e_out_MeV'] = None
        event['cos_scatter'] = None
        event['t_e_MeV'] = round(t_e, 6)
        
        # Photoelectrons are isotropic in the frame of the atom
        # But for high energy photons, they tend forward
        cos_e = np.random.uniform(-1, 1)
        event['cos_electron'] = round(float(cos_e), 6)
        
    elif process == 'pair':
        # Create electron-positron pair
        t_e, t_p = sample_pair_production_energy(E_photon)
        
        event['e_out_MeV'] = None
        event['cos_scatter'] = None
        event['t_e_MeV'] = round(t_e, 6)
        event['t_positron_MeV'] = round(t_p, 6)
        
        # Directions - both tend forward
        cos_e = np.random.uniform(0, 1)  # biased forward
        cos_p = np.random.uniform(0, 1)
        event['cos_electron'] = round(float(cos_e), 6)
        # Note: t_positron already included above

print(f"Generated {len(events)} events")
print("\nFirst 5 events:")
for evt in events[:5]:
    print(evt)
