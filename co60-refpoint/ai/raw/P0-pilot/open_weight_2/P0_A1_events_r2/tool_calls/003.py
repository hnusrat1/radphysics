
import numpy as np
import json

# Physical constants
me = 0.510998950  # electron mass in MeV/c^2

# Mass attenuation coefficients at 2.0 MeV in water (from NIST XCOM)
mu_compton = 0.0428
mu_photoelectric = 0.0001
mu_rayleigh = 0.0002
mu_pair = 0.0052
mu_total = mu_compton + mu_photoelectric + mu_rayleigh + mu_pair

probabilities = [
    mu_compton / mu_total,
    mu_photoelectric / mu_total,
    mu_pair / mu_total,
    mu_rayleigh / mu_total
]

n_events = 200

def sample_events():
    """Generate 200 photon interaction events"""
    np.random.seed(42)
    
    process_counts = np.random.multinomial(n_events, probabilities)
    
    processes = ['compton', 'photoelectric', 'pair', 'rayleigh']
    
    def klein_nishina_sample(E_in):
        """Sample Compton scattering angle from Klein-Nishina distribution"""
        alpha = E_in / me
        
        # Numerical sampling of cos(theta) using rejection method
        for _ in range(1000):
            cos_theta = 2 * np.random.random() - 1.0
            
            # Scattered photon energy
            denom = 1.0 + alpha * (1.0 - cos_theta)
            E_out = E_in / denom
            
            sin2theta = 1.0 - cos_theta**2
            KN_factor = (E_out/E_in)**2 * (E_in/E_out + E_out/E_in - sin2theta)
            
            # Evaluate maximum at several points
            cos_test = np.linspace(-1, 1, 200)
            KN_factors = []
            for c in cos_test:
                d = 1.0 + alpha * (1.0 - c)
                Ep = E_in / d
                s2t = 1.0 - c**2
                KF = (Ep/E_in)**2 * (E_in/Ep + Ep/E_in - s2t)
                KN_factors.append(KF)
            max_KN = np.max(KN_factors)
            
            if np.random.random() * max_KN <= KN_factor:
                return cos_theta, E_out
        
        # Fallback
        return 0.0, E_in / (1.0 + alpha)
    
    def sample_compton_event():
        cos_scatter, E_out = klein_nishina_sample(2.0)
        t_electron = 2.0 - E_out
        
        return {
            "process": "compton",
            "e_in_MeV": 2.0,
            "e_out_MeV": round(E_out, 6),
            "cos_scatter": round(float(cos_scatter), 6),
            "t_e_MeV": round(t_electron, 6),
            "cos_electron": None,
            "t_positron_MeV": None
        }
    
    def sample_photoelectric_event():
        # Photoelectric effect - electron gets full photon energy minus small binding
        # K-shell binding for O is ~0.53 keV, negligible at 2 MeV
        t_electron = 2.0 - 0.00053  # MeV
        
        # At high energies, photoelectrons emitted preferentially forward
        # Distribution approximately proportional to (1 + cosθ)^2 or similar
        # Use exponential distribution peaked forward
        u = np.random.random()
        cos_theta = 1.0 - 2.0 * u / (1.0 + u)  # Approximate forward-peaked distribution
        
        return {
            "process": "photoelectric",
            "e_in_MeV": 2.0,
            "e_out_MeV": None,
            "cos_scatter": None,
            "t_e_MeV": round(t_electron, 6),
            "cos_electron": round(float(np.clip(cos_theta, -1, 1)), 6),
            "t_positron_MeV": None
        }
    
    def sample_pair_production_event():
        threshold = 2.0 * me  # 1.0219979 MeV
        excess = 2.0 - threshold  # Available kinetic energy
        
        # Energy sharing: roughly uniform for high energies, but slightly forward
        fraction_pos = np.random.uniform(0.35, 0.65)
        
        T_pos = fraction_pos * excess
        T_ele = excess - T_pos
        
        # Angular distributions: both particles emitted somewhat forward
        # With some back-to-back correlation
        cos_pos = np.random.beta(1.5, 0.5) * 2 - 1  # Forward skewed
        cos_ele = np.random.beta(1.5, 0.5) * 2 - 1
        
        return {
            "process": "pair",
            "e_in_MeV": 2.0,
            "e_out_MeV": None,
            "cos_scatter": None,
            "t_e_MeV": round(T_ele, 6),
            "cos_electron": round(float(cos_ele), 6),
            "t_positron_MeV": round(T_pos, 6)
        }
    
    def sample_rayleigh_event():
        # Coherent (Rayleigh) scattering - elastic, no energy loss
        # Cross section peaks at small angles (forward)
        # Use form factor-modified distribution
        
        # For 2 MeV photons, mainly very forward scattering
        cos_theta = 1.0 - np.random.exponential(scale=0.05)
        cos_theta = min(max(cos_theta, 0.0), 1.0)
        
        return {
            "process": "rayleigh",
            "e_in_MeV": 2.0,
            "e_out_MeV": 2.0,
            "cos_scatter": round(float(cos_theta), 6),
            "t_e_MeV": None,
            "cos_electron": None,
            "t_positron_MeV": None
        }
    
    # Generate events per process
    events = []
    for i in range(process_counts[0]):
        events.append(sample_compton_event())
    for i in range(process_counts[1]):
        events.append(sample_photoelectric_event())
    for i in range(process_counts[2]):
        events.append(sample_pair_production_event())
    for i in range(process_counts[3]):
        events.append(sample_rayleigh_event())
    
    # Shuffle
    np.random.shuffle(events)
    
    return events

events = sample_events()

print(f"Total events: {len(events)}")
print("\nProcess distribution:")
from collections import Counter
counts = Counter(ev['process'] for ev in events)
for proc, count in sorted(counts.items()):
    print(f"  {proc}: {count}")

# Verify structure
print("\nSample event structure:")
print(json.dumps(events[0], indent=2))
