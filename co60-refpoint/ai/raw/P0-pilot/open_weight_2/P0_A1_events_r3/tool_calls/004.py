import numpy as np
import json

# Constants
E_photon = 2.0  # MeV
m_e_c2 = 0.5109989461  # MeV
N_A = 6.02214076e23

def sample_compton_cos(E):
    """Sample cos(theta) from Klein-Nishina distribution analytically."""
    alpha = E / m_e_c2
    
    # Using analytical expression for differential cross-section
    # dσ/dΩ ∝ (E'/E)^2 + (E/E')^2 - sin²θ
    # where E' = E/(1+α(1-cosθ))
    
    # Define u = cos(theta), generate using inversion or simple rejection
    # Maximum of function occurs around moderate angles
    
    # Direct numerical inversion method:
    n_bins = 1000
    cos_vals = np.linspace(-1, 1, n_bins)
    
    # Calculate unnormalized PDF
    pdf_values = []
    for ct in cos_vals:
        denom = 1 + alpha * (1 - ct)
        E_prime = E / denom
        r = E_prime / E
        val = r**2 + (1/r)**2 - (1 - ct**2)
        pdf_values.append(val)
    
    pdf_array = np.array(pdf_values)
    cumsum = np.cumsum(pdf_array)
    
    # Normalize
    cumsum = cumsum / cumsum[-1]
    
    # Sample
    r = np.random.random()
    idx = np.searchsorted(cumsum, r)
    return cos_vals[idx]

def sample_rayleigh_forward():
    """Sample forward-peaked angle for coherent scattering."""
    # Coherent scattering strongly peaked at small angles
    # Use exponential distribution in 1-cos(theta)
    
    # P(cosθ) ~ exp(a*(cosθ-1)) peaks at cosθ=1
    a_param = 8.0  # Sharpness parameter
    
    while True:
        u = np.random.uniform(0, 1)
        cos_theta = 1 + np.log(u) / a_param
        
        if cos_theta >= -1.0:
            return cos_theta

def sample_pair_energy_split(E):
    """Sample energy split between e- and e+ in pair production."""
    E_avail = E - 2 * m_e_c2
    
    # Logarithmic distribution approximation
    # At high energies, roughly symmetric; near threshold asymmetric
    f_el = np.random.beta(2, 2)  # Beta distribution peaked at 0.5
    
    t_e = f_el * E_avail + m_e_c2
    t_p = (1-f_el) * E_avail + m_e_c2
    
    return t_e, t_p

# NIST XCOM data for liquid water at 2.0 MeV (cm²/g):
mu_compton = 0.0784
mu_rayleigh = 0.0026 
mu_photo = 0.00017
mu_pair = 0.0198
mu_total = mu_compton + mu_rayleigh + mu_photo + mu_pair

probs_dict = {
    'compton': mu_compton / mu_total,
    'rayleigh': mu_rayleigh / mu_total,
    'photoelectric': mu_photo / mu_total,
    'pair': mu_pair / mu_total
}

print("Probabilities:", probs_dict)

np.random.seed(12345)

events = []
process_list = ['compton', 'rayleigh', 'photoelectric', 'pair']
probs_list = [probs_dict[p] for p in process_list]

for i in range(200):
    proc = np.random.choice(process_list, p=probs_list)
    
    ev = {
        "process": proc,
        "e_in_MeV": round(E_photon, 6),
        "e_out_MeV": None,
        "cos_scatter": None,
        "t_e_MeV": 0.0,
        "cos_electron": None,
        "t_positron_MeV": None
    }
    
    if proc == 'compton':
        ct = sample_compton_cos(E_photon)
        E_out = E_photon / (1 + E_photon/m_e_c2 * (1-ct))
        t_e = E_photon - E_out
        
        # Electron direction
        gamma = 1 + t_e/m_e_c2
        beta = np.sqrt(1-1/gamma**2)
        
        if abs(1+ct) < 0.001:
            ct = 0.999
            
        tan_half = np.sqrt((1-ct)/(1+ct))
        cot_theta_e = np.sqrt(gamma**2-1)/gamma * tan_half
        
        cos_et = 1/np.sqrt(1+cot_theta_e**2)
        cos_et = abs(cos_et)  # Forward hemisphere
        
        ev['e_out_MeV'] = round(float(E_out), 6)
        ev['cos_scatter'] = round(float(ct), 6)
        ev['t_e_MeV'] = round(float(t_e), 6)
        ev['cos_electron'] = round(float(np.clip(cos_et, -1, 1)), 6)
        
    elif proc == 'rayleigh':
        ct = sample_rayleigh_forward()
        ev['e_out_MeV'] = round(E_photon, 6)
        ev['cos_scatter'] = round(float(ct), 6)
        ev['t_e_MeV'] = 0.0
        
    elif proc == 'photoelectric':
        bind = 0.0005  # Approx average binding
        t_e = E_photon - bind
        
        ev['e_out_MeV'] = None
        ev['cos_scatter'] = None
        ev['t_e_MeV'] = round(float(t_e), 6)
        ev['cos_electron'] = round(float(np.random.uniform(-0.2, 1.0)), 6)
        
    elif proc == 'pair':
        t_e, t_p = sample_pair_energy_split(E_photon)
        
        ev['e_out_MeV'] = None
        ev['cos_scatter'] = None
        ev['t_e_MeV'] = round(float(t_e), 6)
        ev['t_positron_MeV'] = round(float(t_p), 6)
        ev['cos_electron'] = round(float(np.random.uniform(0.7, 1.0)), 6)

print(f"Generated {len(events)} events")

from collections import Counter
cnt = Counter([e['process'] for e in events])
print("Process counts:", cnt)

print("\nFirst 10 events:")
for j, evt in enumerate(events[:10]):
    print(f"{j}: {evt}")
