import numpy as np
import json

# Constants
me = 0.5109989461
r_e = 2.8179403227e-13
NA = 6.02214076e23
M_H2O = 18.015
N_e = 10
E1 = 1.1732
E2 = 1.3325

np.random.seed(12345)

# Cross sections
def sigma_KN(E):
    k = E / me
    term1 = ((1+k)/(k*k)) * (2*(1+k)/(1+2*k) - np.log(1+2*k)/k)
    term2 = np.log(1+2*k)/(2*k)
    term3 = (1+3*k)/(1+2*k)**2
    return 2 * np.pi * r_e**2 * (term1 + term2 - term3)

def partial_mu(E):
    mu_comp = N_e * sigma_KN(E) * NA / M_H2O
    mu_rayl = 0.0014 * (1.0 / max(E, 0.01))**2
    mu_pe = 0.0016 * (1.0 / max(E, 0.01))**3.5
    if E > 1.022:
        mu_pair = 0.00087 * (E - 1.022) / (1.25 - 1.022)
    else:
        mu_pair = 0.0
    mu_tot = mu_comp + mu_rayl + mu_pe + mu_pair
    return mu_tot, mu_comp, mu_rayl, mu_pe, mu_pair

# Sampling functions
def sample_compton_cos(E):
    alpha = E / me
    while True:
        mu = np.random.uniform(-1, 1)
        f = 1.0 / (1.0 + alpha * (1.0 - mu))
        dcs = f**2 * (f + 1.0/f - (1.0 - mu**2))
        if np.random.random() < dcs / 2.0:
            return mu

def compton_electron_cos(E, cos_theta):
    sin_theta = np.sqrt(1 - cos_theta**2)
    tan_half = sin_theta / (1 + cos_theta)
    alpha = E / me
    cot_phi = (1 + alpha) * tan_half
    if cot_phi == 0:
        return 0.0
    tan_phi = 1.0 / cot_phi
    cos_phi = 1.0 / np.sqrt(1.0 + tan_phi**2)
    return cos_phi

def sample_rayleigh_cos():
    return np.random.uniform(0.999, 1.0)

def sample_pe_electron_cos(E_kin):
    E_tot = E_kin + me
    beta = np.sqrt(1 - (me/E_tot)**2)
    # Find max
    mus = np.linspace(-1, 1, 1001)
    vals = (1 - mus**2) / (1 - beta*mus)**4
    max_val = np.max(vals)
    while True:
        mu = np.random.uniform(-1, 1)
        val = (1 - mu**2) / (1 - beta*mu)**4
        if np.random.random() < val / max_val:
            return mu

def sample_pair(K_avail):
    t_e = np.random.uniform(0, K_avail)
    t_p = K_avail - t_e
    cos_e = np.random.uniform(0.98, 1.0)
    cos_p = np.random.uniform(0.98, 1.0)
    return t_e, t_p, cos_e, cos_p

# Generate 200 events
events = []

# Fraction of scattered interactions (approximate)
# We'll set it to 0.35 (35% scattered, 65% primary)
scattered_frac = 0.35

while len(events) < 200:
    # Sample incident type
    is_scattered = np.random.random() < scattered_frac
    
    if not is_scattered:
        # Primary photon
        E_in = E1 if np.random.random() < 0.5 else E2
    else:
        # Scattered photon: sample energy from a distribution
        # Use a simple bimodal: 70% chance low energy (0.01-0.3), 30% chance medium (0.3-1.33)
        u = np.random.random()
        if u < 0.7:
            E_in = np.random.uniform(0.01, 0.3)
        else:
            E_in = np.random.uniform(0.3, 1.3325)
    
    # Sample process
    mt, mc, mr, mp, mpr = partial_mu(E_in)
    p_comp = mc / mt
    p_rayl = mr / mt
    p_pe = mp / mt
    # Note: p_pair is mpr/mt
    
    xi = np.random.random()
    if xi < p_comp:
        process = 'compton'
    elif xi < p_comp + p_rayl:
        process = 'rayleigh'
    elif xi < p_comp + p_rayl + p_pe:
        process = 'photoelectric'
    else:
        process = 'pair'
    
    # Build event
    ev = {'process': process, 'e_in_MeV': E_in,
          'e_out_MeV': None, 'cos_scatter': None,
          't_e_MeV': 0.0, 'cos_electron': None, 't_positron_MeV': None}
    
    if process == 'compton':
        cos_s = sample_compton_cos(E_in)
        E_out = E_in / (1 + (E_in/me) * (1 - cos_s))
        t_e = E_in - E_out
        cos_e = compton_electron_cos(E_in, cos_s)
        ev['e_out_MeV'] = E_out
        ev['cos_scatter'] = cos_s
        ev['t_e_MeV'] = t_e
        ev['cos_electron'] = cos_e
    elif process == 'rayleigh':
        cos_s = sample_rayleigh_cos()
        ev['e_out_MeV'] = E_in
        ev['cos_scatter'] = cos_s
        # t_e_MeV stays 0.0
        # cos_electron stays None
    elif process == 'photoelectric':
        t_e = E_in  # neglect binding energy
        cos_e = sample_pe_electron_cos(t_e)
        ev['t_e_MeV'] = t_e
        ev['cos_electron'] = cos_e
    else:  # pair
        K_avail = E_in - 2*me
        if K_avail <= 0:
            # Should not happen because we only sample pair when E>1.022
            continue
        t_e, t_p, cos_e, cos_p = sample_pair(K_avail)
        ev['t_e_MeV'] = t_e
        ev['cos_electron'] = cos_e
        ev['t_positron_MeV'] = t_p
    
    events.append(ev)

print(f"Generated {len(events)} events")

# Check distribution of processes
processes = [ev['process'] for ev in events]
print("Process distribution:")
from collections import Counter
cnt = Counter(processes)
for p, c in cnt.most_common():
    print(f"  {p}: {c}")

# Print first 3 events
print("\nFirst 3 events:")
for ev in events[:3]:
    print(json.dumps(ev, indent=2))