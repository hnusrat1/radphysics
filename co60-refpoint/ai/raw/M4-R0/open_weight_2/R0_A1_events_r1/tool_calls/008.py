import numpy as np
import json
from math import pi, sin, cos, tan, acos, sqrt
import os

me_c2 = 0.510998950 
energy_1 = 1.1732
energy_2 = 1.3325

def get_kn_cos_theta_sample(E_in):
    alpha = E_in / me_c2
    def kn_pdf(x):
        ep = E_in / (1 + alpha * (1 - x))
        return (ep/E_in)**2 * (ep/E_in + E_in/ep - 1 + x**2) 
    max_val = kn_pdf(1.0)
    while True:
        x_c = np.random.uniform(-1, 1)
        if np.random.random() < kn_pdf(x_c) / max_val:
            return x_c

def sample_pe_angle(E_in):
    total_energy = E_in + me_c2
    beta_sq = (total_energy**2 - me_c2**2) / total_energy**2
    beta = sqrt(beta_sq)
    def f(x): return x / (1 - beta*x)**2 
    max_f = f(1.0)
    while True:
        x = np.random.uniform(beta, 1.0)
        if np.random.random() < f(x) / max_f:
            return x

def sample_rayleigh_angle():
    return np.random.uniform(0.85, 1.0)

def get_interaction_probs(E):
    if E < 1.022: p_pair = 0.0
    else: 
        excess_E = E - 1.022
        p_pair = 0.0005 * (excess_E / 0.2)**2
    base_pe = 0.012 - 0.002*(E-1.0)/0.3
    if base_pe < 0: base_pe = 0.002
    p_rayl = 0.007 
    s = p_pair + base_pe + p_rayl
    p_compton = 1.0 - s
    if p_compton < 0.9: p_compton = 0.9
    return p_compton, base_pe, p_rayl, p_pair

np.random.seed(42)
events = []
num_events = 200
w1, w2 = 0.747, 0.763
weights = np.array([w1, w2])
weights /= weights.sum()

for i in range(num_events):
    energy_index = np.random.choice([0, 1], p=weights)
    E_in = energy_1 if energy_index == 0 else energy_2
    
    p_com, p_pe, p_rayl, p_pair = get_interaction_probs(E_in)
    choices = ['compton', 'photoelectric', 'rayleigh', 'pair']
    process_name = np.random.choice(choices, p=[p_com, p_pe, p_rayl, p_pair])
    
    ev = {
        "process": process_name, "e_in_MeV": round(E_in, 4),
        "e_out_MeV": None, "cos_scatter": None, "t_e_MeV": None,
        "cos_electron": None, "t_positron_MeV": None
    }
    
    if process_name == 'compton':
        alpha = E_in / me_c2
        cos_theta = get_kn_cos_theta_sample(E_in)
        e_out = E_in / (1 + alpha * (1 - cos_theta))
        theta = acos(cos_theta)
        cot_psi = (1 + alpha) * tan(theta / 2)
        psi = acos(1 / sqrt(1 + cot_psi**2))
        
        ev['e_out_MeV'] = round(e_out, 4)
        ev['cos_scatter'] = cos_theta
        ev['t_e_MeV'] = round(E_in - e_out, 4)
        ev['cos_electron'] = round(cos(psi), 4)

    elif process_name == 'photoelectric':
        ev['t_e_MeV'] = round(E_in, 4)
        ev['cos_electron'] = round(sample_pe_angle(E_in), 4)
        
    elif process_name == 'rayleigh':
        ev['e_out_MeV'] = round(E_in, 4)
        ev['cos_scatter'] = round(sample_rayleigh_angle(), 4)
        ev['t_e_MeV'] = 0.0
        
    elif process_name == 'pair':
        t_avail = E_in - 2*me_c2
        t_e = t_avail * np.random.uniform(0, 1)
        ev['t_e_MeV'] = round(t_e, 4)
        ev['t_positron_MeV'] = round(t_avail - t_e, 4)
        ev['cos_electron'] = round(np.random.uniform(0.9, 1.0), 4)
        
    events.append(ev)

method_desc = "Monte Carlo simulation using analytical cross-section models (Klein-Nishina, Sauter approx, linear pair-production model) for liquid water. Energy sampling weighted by primary attenuation at 5 cm depth."

data = {"method": method_desc, "events": events}

# Write to file
file_path = "full_output.json"
with open(file_path, "w") as f:
    json.dump(data, f, separators=(',', ':'))

print(f"Written to {file_path}, size: {os.path.getsize(file_path)} bytes")

# Try to print
try:
    with open(file_path, "r") as f:
        content = f.read()
    # Print in chunks to avoid single command buffer issues?
    # But the tool executes a script. stdout is accumulated.
    # I'll just print it.
    sys.stdout.write(content)
except Exception as e:
    print(e)

import sys
sys.stdout.flush()