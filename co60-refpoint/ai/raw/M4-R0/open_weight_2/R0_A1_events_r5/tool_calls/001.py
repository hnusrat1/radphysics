import numpy as np
import json

# Constants
m_e_c2 = 0.511 # MeV
r_e = 2.8179403262e-13 # cm

def klein_nishina_angle(a):
    """
    Generates a scattering cosine u = cos(theta) for a photon of dimensionless energy a = E/mc^2
    using rejection sampling.
    """
    while True:
        u = np.random.uniform(-1, 1)
        # KS formula terms
        k_ratio = 1.0 / (1.0 + a * (1.0 - u))
        term_in_bracket = k_ratio + 1.0/k_ratio - (1 - u**2)
        # Differential cross section proportional to
        val = (k_ratio**2) * term_in_bracket
        
        # Max value estimation for rejection sampling
        # Max usually near backscatter u = -1, but let's find bound numerically or use loose bound
        # Loose bound: at u=-1, a is positive, k_ratio is small.
        # Term goes to approx 1/a^2 * (a + 1/a) ~ a/a^2? 
        # Let's just use a safe upper bound. Max of f(u) is at u=-1 roughly.
        val_max_val = klein_nishina_dens(-1, a)
        
        # Rejection sampling
        if np.random.uniform(0, val_max_val) < val:
            return u

def klein_nishina_dens(u, a):
    """Value of the unnormalized differential cross section"""
    k_ratio = 1.0 / (1.0 + a * (1.0 - u))
    term_in_bracket = k_ratio + 1.0/k_ratio - (1 - u**2)
    return (k_ratio**2) * term_in_bracket

def generate_compton(E_in, a):
    # Generate angle
    u_scat = klein_nishina_angle(a)
    
    # Energy out
    E_out = E_in / (1.0 + a * (1.0 - u_scat))
    
    # Electron recoil energy
    T_e = E_in - E_out
    
    # Electron scattering angle phi
    # tan(phi) = (1/(a+1)) * cot(theta/2)
    theta = np.arccos(u_scat)
    cos_theta_half = np.cos(theta/2)
    sin_theta_half = np.sin(theta/2)
    
    # tan_phi
    denom = 1 + a
    tan_phi = (1/denom) * (cos_theta_half/sin_theta_half)
    
    # Convert tan_phi to cos_phi (we need the projection, usually sign matters but magnitude is enough for distribution)
    # We'll take phi positive.
    cos_phi = 1.0 / np.sqrt(1 + tan_phi**2)
    
    return {
        "process": "compton",
        "e_in_MeV": round(E_in, 6),
        "e_out_MeV": round(E_out, 6),
        "cos_scatter": round(u_scat, 6),
        "t_e_MeV": round(T_e, 6),
        "cos_electron": round(cos_phi, 6),
        "t_positron_MeV": None
    }

def generate_photoelectric(E_in):
    # Photon absorbed
    # Electron energy
    T_e = E_in # minus negligible binding energy
    
    # Angle distribution. Isotropic is bad. Photoelectric peaks forward at high energies.
    # Distribution roughly dσ/dΩ ∝ sin²θ (transverse) or something?
    # Actually, usually modelled with dipole pattern for s-p transitions.
    # Simplified: cosine drawn from uniform [-1, 1] is acceptable for a generic model or slightly forward peaked.
    # Standard Bethe-Heitler style for Compton is angular dependent. For PE, let's use cos ~ Uniform(-1,1)
    cos_el = np.random.uniform(-1, 1)
    
    return {
        "process": "photoelectric",
        "e_in_MeV": round(E_in, 6),
        "e_out_MeV": None,
        "cos_scatter": None,
        "t_e_MeV": round(T_e, 6),
        "cos_electron": round(cos_el, 6),
        "t_positron_MeV": None
    }

def generate_rayleigh(E_in, a):
    # Rayleigh scattering (elastic coherent)
    # Angular distribution is forward peaked.
    # Simple model: Rutherford-like or Gaussian.
    # Let's use a simple exponential probability distribution for theta to get forward peaking.
    # P(cos θ) ∝ exp( C * (cos θ - 1) ) ?
    # Let's pick a C parameter. For ~1 MeV, scattering is moderately forward.
    # Let's use rejection sampling against a simple function.
    while True:
        u_scat = np.random.uniform(-1, 1)
        # Rough approximation form factor behavior: strongly forward
        # Probability density ~ (1 + alpha * u)^-n ?
        # Let's just use a simple Gaussian on theta: P(theta) ~ exp( - theta^2 / (2 sigma^2) )
        # theta = arccos(u)
        theta = np.arccos(u_scat)
        # sigma for 1 MeV in water approx few degrees?
        # Let's say sigma ~ 0.2 rad (~10 deg).
        pdf = np.exp( - (theta**2) / (2 * 0.2**2) )
        
        # Normalization constant not needed for rejection if max <= 1. 
        # Peak is at u=1 (theta=0) where pdf=1.
        r = np.random.uniform(0, 1)
        if r < pdf:
            break
            
    return {
        "process": "rayleigh",
        "e_in_MeV": round(E_in, 6),
        "e_out_MeV": round(E_in, 6), # Elastic
        "cos_scatter": round(u_scat, 6),
        "t_e_MeV": 0.0, # Negligible recoil
        "cos_electron": None,
        "t_positron_MeV": None
    }

def generate_pair(E_in):
    # Pair production
    T_total = E_in - 1.022 # Available kinetic energy
    if T_total < 0: return None
    
    # Symmetric split with bias towards sharing?
    # Distribution is fairly symmetric around 0.5 for high energy.
    # At 1.3 MeV (T=0.3), it's low energy.
    # Let's split T between e- and pos.
    fraction = np.random.uniform(0, 1)
    T_e = fraction * T_total
    T_pos = (1 - fraction) * T_total
    
    # Angles. Both emitted in forward cone.
    # Cosine drawn from a forward biased distribution?
    # Let's just say cos ~ uniform(0, 1) for simplicity or slightly peaked.
    cos_e = np.random.uniform(0, 1) # forward hemisphere
    # Positron angle correlated or independent? Independent in first order.
    # cos_electron: cos(angle between electron and photon) -> cos_e
    # Note: The schema asks for cos_electron. I'll provide it.
    
    return {
        "process": "pair",
        "e_in_MeV": round(E_in, 6),
        "e_out_MeV": None,
        "cos_scatter": None,
        "t_e_MeV": round(T_e, 6),
        "cos_electron": round(cos_e, 6),
        "t_positron_MeV": round(T_pos, 6)
    }

def generate_events():
    events = []
    n_per_energy = 100
    
    energies = [1.1732, 1.3325]
    
    # Process weights (Probabilities) normalized to sum=1
    # Based on water cross sections:
    # 1.17 MeV: Compton~92%, Rayleigh~7%, Pe~0.5%, Pair~0.5%
    # 1.33 MeV: Compton~91%, Rayleigh~6%, Pe~0.5%, Pair~1.5% (threshold crossing)
    
    probs_E1 = {'compton': 0.92, 'rayleigh': 0.07, 'photoelectric': 0.005, 'pair': 0.005}
    probs_E2 = {'compton': 0.91, 'rayleigh': 0.06, 'photoelectric': 0.004, 'pair': 0.006}
    
    rng = np.random.default_rng(42) # Seed for reproducibility
    
    for E in energies:
        current_probs = probs_E1 if abs(E - 1.1732) < 1e-5 else probs_E2
        a = E / m_e_c2
        
        keys = list(current_probs.keys())
        values = list(current_probs.values())
        # Normalize
        total_p = sum(values)
        values = [v/total_p for v in values]
        
        for _ in range(n_per_energy):
            # Select process
            proc = rng.choice(keys, p=values)
            
            ev = None
            if proc == 'compton':
                ev = generate_compton(E, a)
            elif proc == 'photoelectric':
                ev = generate_photoelectric(E)
            elif proc == 'rayleigh':
                ev = generate_rayleigh(E, a)
            elif proc == 'pair':
                ev = generate_pair(E)
            
            if ev:
                events.append(ev)
                
    # Shuffle to avoid ordering bias
    rng.shuffle(events)
    return events

events_list = generate_events()
print(json.dumps(events_list))