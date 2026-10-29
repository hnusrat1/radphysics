
import numpy as np
import json

# Physical constants
me = 0.510998950  # electron mass in MeV/c^2
c_eV = 510998.950  # electron mass in eV

# Mass attenuation coefficients at 2.0 MeV in water (from NIST XCOM)
mu_compton = 0.0428
mu_photoelectric = 0.0001
mu_rayleigh = 0.0002
mu_pair = 0.0052
mu_total = mu_compton + mu_photoelectric + mu_rayleigh + mu_pair

# Event counts for 200 events
np.random.seed(42)
n_events = 200

# Determine number of events per process (using multinomial sampling)
processes = ['compton', 'photoelectric', 'pair', 'rayleigh']
probabilities = [
    mu_compton / mu_total,
    mu_photoelectric / mu_total,
    mu_pair / mu_total,
    mu_rayleigh / mu_total
]

process_counts = np.random.multinomial(n_events, probabilities)
print(f"Process counts: {dict(zip(processes, process_counts))}")

events = []

def klein_nishina_differential(sigma, theta):
    """Generate scattering angles from Klein-Nishina distribution"""
    # For Compton scattering at energy E, the differential cross section is proportional to:
    # dσ/dΩ ∝ (E'/E)^2 * (E/E' + E'/E - sin^2θ)
    # where E' = E / (1 + (E/me c^2)(1-cosθ))
    
    # We use rejection sampling
    alpha = sigma / me  # incident photon energy in units of mc^2
    
    max_ratio = 1.0 + 1.0/(1.0 + 2*alpha) - alpha * np.sin(np.arccos((1.0 + alpha - np.sqrt((1.0 + alpha)**2 - 1.0))) / (2*alpha + 1.0)) if True else 1.0
    
    while True:
        cos_theta = 2 * np.random.random() - 1.0
        
        # Calculate scattered photon energy ratio
        denom = 1.0 + alpha * (1.0 - cos_theta)
        ratio_E_prime_E = 1.0 / denom
        E_over_Ep = denom
        
        # Klein-Nishina formula (simplified angular distribution)
        # F(cosθ) = (E'/E)^2 * (E/E' + E'/E - sin^2θ)
        sin2theta = 1.0 - cos_theta**2
        f_kn = ratio_E_prime_E**2 * (E_over_Ep + ratio_E_prime_E - sin2theta)
        
        # Maximum value of F occurs at cosθ = ±1
        f_max = 1.0 + 1.0/((1 + 2*alpha)**2) - alpha*(4*alpha)/(1 + 2*alpha) if alpha > 0 else 1.0
        # Actually let's compute maximum numerically
        cos_vals = np.linspace(-1, 1, 1000)
        f_vals = np.array([
            (1/(1 + alpha*(1-c))**2) * ((1 + alpha*(1-c)) + (1/(1 + alpha*(1-c))) - (1 - c**2))
            for c in cos_vals
        ])
        f_max = np.max(f_vals)
        
        u = np.random.random() * f_max
        if u <= f_kn:
            return cos_theta


def sample_compton_event(E_in=2.0):
    """Sample a Compton scattering event"""
    alpha = E_in / me  # energy in units of electron rest mass
    
    # Sample scattering angle from Klein-Nishina distribution
    # Using numerical inverse transform or rejection
    cos_theta = 0.0
    attempts = 0
    while attempts < 10000:
        cos_trial = 2 * np.random.random() - 1.0
        
        # Scattered photon energy
        E_out = E_in / (1.0 + alpha * (1.0 - cos_trial))
        
        # Klein-Nishina differential cross section (unnormalized)
        sin2theta = 1.0 - cos_trial**2
        KN_factor = (E_out/E_in)**2 * (E_in/E_out + E_out/E_in - sin2theta)
        
        # Maximum KN factor
        cos_test = np.linspace(-1, 1, 500)
        KN_factors = [(E_in/(1.0 + alpha*(1-c)))**2 * 
                     ((1.0 + alpha*(1-c)) + (1.0/(1.0 + alpha*(1-c))) - (1-c**2)) 
                     for c in cos_test]
        max_KN = max(KN_factors)
        
        if np.random.random() * max_KN <= KN_factor:
            cos_theta = cos_trial
            break
        attempts += 1
    
    if attempts == 10000:
        # Fallback
        cos_theta = -0.5
    
    E_out = E_in / (1.0 + alpha * (1.0 - cos_theta))
    t_electron = E_in - E_out
    
    return {
        "process": "compton",
        "e_in_MeV": round(E_in, 10),
        "e_out_MeV": round(E_out, 10),
        "cos_scatter": round(float(cos_theta), 10),
        "t_e_MeV": round(t_electron, 10),
        "cos_electron": None,
        "t_positron_MeV": None
    }


def sample_photoelectric_event(E_in=2.0):
    """Sample a photoelectric absorption event"""
    # Binding energy is small (~keV range) compared to 2 MeV, neglect it
    # Or include K-shell binding energy of oxygen (~0.53 keV)
    binding_energy_O = 0.00053  # MeV
    t_electron = E_in - binding_energy_O
    
    # Angular distribution: preferentially forward at high energies
    # Use approximate distribution ~ (1 + cosθ) or similar
    cos_theta = np.random.exponential(scale=1.0/10.0)  # Forward-peaked
    cos_theta = min(max(cos_theta, -0.5), 1.0)  # Clip reasonable range
    
    return {
        "process": "photoelectric",
        "e_in_MeV": round(E_in, 10),
        "e_out_MeV": None,
        "cos_scatter": None,
        "t_e_MeV": round(t_electron, 10),
        "cos_electron": round(float(cos_theta), 10),
        "t_positron_MeV": None
    }


def sample_pair_production_event(E_in=2.0):
    """Sample a pair production event"""
    threshold = 2.0 * me  # 1.022 MeV
    excess_energy = E_in - threshold
    
    # Energy sharing between positron and electron
    # Roughly uniform or slightly peaked
    fraction_pos = np.random.uniform(0.3, 0.7)  # Positron gets this fraction of excess energy
    
    # Relativistic kinetic energy
    T_pos_e = fraction_pos * excess_energy
    T_elec = (1.0 - fraction_pos) * excess_energy
    
    # For pair production near threshold, both particles are emitted roughly back-to-back
    # At higher energies, forward-peaked
    cos_pos = np.random.beta(a=1.0, b=0.1) * 2.0 - 0.5  # Forward skewed
    cos_ele = -cos_pos * np.random.uniform(0.7, 1.0)  # Opposite direction but some spread
    
    return {
        "process": "pair",
        "e_in_MeV": round(E_in, 10),
        "e_out_MeV": None,
        "cos_scatter": None,
        "t_e_MeV": round(T_elec, 10),
        "cos_electron": round(float(cos_ele), 10),
        "t_positron_MeV": round(T_pos_e, 10)
    }


def sample_rayleigh_event(E_in=2.0):
    """Sample a Rayleigh (coherent) scattering event"""
    # Elastic scattering - no energy loss
    E_out = E_in
    
    # Angular distribution follows atomic form factor
    # Preferentially forward (small angles) at high energies
    # Use sin^2 θ * F^2(q) behavior
    # Simple approximation: mostly forward scattering
    
    cos_theta = 1.0 - np.random.exponential(scale=0.1)
    cos_theta = min(max(cos_theta, 0.0), 1.0)
    
    # Small but non-zero angle
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    if sin_theta < 0.01:
        cos_theta = 0.99
    
    return {
        "process": "rayleigh",
        "e_in_MeV": round(E_in, 10),
        "e_out_MeV": round(E_out, 10),
        "cos_scatter": round(float(cos_theta), 10),
        "t_e_MeV": None,
        "cos_electron": None,
        "t_positron_MeV": None
    }


# Generate events
for _ in range(process_counts[0]):
    events.append(sample_compton_event())

for _ in range(process_counts[1]):
    events.append(sample_photoelectric_event())

for _ in range(process_counts[2]):
    events.append(sample_pair_production_event())

for _ in range(process_counts[3]):
    events.append(sample_rayleigh_event())

# Shuffle to randomize order
np.random.shuffle(events)

print(f"Generated {len(events)} events")
print(f"First 5 events:")
for i, ev in enumerate(events[:5]):
    print(f"{i}: {ev}")
