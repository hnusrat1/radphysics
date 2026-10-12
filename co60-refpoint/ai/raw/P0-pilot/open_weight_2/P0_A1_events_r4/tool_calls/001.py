import numpy as np
import json

def generate_events(n=200, seed=42):
    np.random.seed(seed)
    
    # Cross section ratios for water at 2.0 MeV (approx from NIST XCOM)
    weights = {
        'compton': 0.9625,
        'rayleigh': 0.0268,
        'photoelectric': 0.0102,
        'pair': 0.0005
    }
    proc_names = list(weights.keys())
    proc_probs = np.array(list(weights.values()))
    proc_probs /= proc_probs.sum()
    
    E0 = 2.0 # MeV
    mc2 = 0.510998 # MeV
    
    events = []
    
    for i in range(n):
        proc = np.random.choice(proc_names, p=proc_probs)
        
        evt = {'process': proc, 'e_in_MeV': E0}
        
        if proc == 'compton':
            # Klein-Nishina sampling via rejection
            def kn_pdf(x): # x = cos_theta
                E_p = E0 / (1 + (E0/mc2)*(1-x))
                term = (E_p/E0)**2 * (E_p/E0 + E0/E_p - (1-x**2))
                return term
            
            # Max of KN PDF is at x=1 (forward)
            max_val = kn_pdf(1.0)
            
            cos_sc = 0.0
            while True:
                c = 2*np.random.rand()-1 # uniform in [-1, 1]
                u = np.random.rand()
                if u * max_val <= kn_pdf(c):
                    cos_sc = c
                    break
            
            E_out = E0 / (1 + (E0/mc2)*(1-cos_sc))
            t_e = E0 - E_out
            
            # Electron angle: cot(theta_e) = (1 + E0/mc2) * tan(theta_sc/2)
            theta_sc = np.arccos(cos_sc)
            cos_theta_e = 1.0 / np.sqrt(1 + ((1+E0/mc2)*np.tan(theta_sc/2))**2)
            
            evt['e_out_MeV'] = round(E_out, 6)
            evt['cos_scatter'] = round(cos_sc, 6)
            evt['t_e_MeV'] = round(t_e, 6)
            evt['cos_electron'] = round(cos_theta_e, 6)
            evt['t_positron_MeV'] = None
            
        elif proc == 'photoelectric':
            # Electron gets almost all energy (binding energy neglected at 2 MeV)
            t_e = E0
            # Highly forward peaked
            cos_e = np.random.uniform(0.95, 1.0)
            
            evt['e_out_MeV'] = None
            evt['cos_scatter'] = None
            evt['t_e_MeV'] = round(t_e, 6)
            evt['cos_electron'] = round(cos_e, 6)
            evt['t_positron_MeV'] = None
            
        elif proc == 'rayleigh':
            # Coherent elastic scattering
            E_out = E0
            t_e = 0.0
            # Forward peaked, sample from exp distribution for (1-cos)
            delta = -np.log(np.random.rand()*0.9 + 0.1) * 0.05 # scale ~0.05
            cos_sc = np.cos(delta)
            
            evt['e_out_MeV'] = round(E_out, 6)
            evt['cos_scatter'] = round(float(cos_sc), 6)
            evt['t_e_MeV'] = round(t_e, 6)
            evt['cos_electron'] = None
            evt['t_positron_MeV'] = None
            
        elif proc == 'pair':
            # Pair production
            T_total = E0 - 1.022
            # Roughly symmetric sharing at 2 MeV
            f_e = np.random.uniform(0.3, 0.7)
            t_e = T_total * f_e
            t_pos = T_total * (1-f_e)
            
            # Forward angles
            cos_e = np.random.uniform(0.97, 1.0)
            cos_pos = np.random.uniform(0.97, 1.0) # not used in schema but good to know
            
            evt['e_out_MeV'] = None
            evt['cos_scatter'] = None
            evt['t_e_MeV'] = round(t_e, 6)
            evt['cos_electron'] = round(cos_e, 6)
            evt['t_positron_MeV'] = round(t_pos, 6)
            
        events.append(evt)
        
    return events

events = generate_events()
print(json.dumps(events, indent=2))