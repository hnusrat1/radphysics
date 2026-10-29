import numpy as np
import json

def generate_events(n=200, seed=123):
    np.random.seed(seed)
    
    # Cross section ratios for liquid water at 2.0 MeV (NIST XCOM approx.)
    # Compton dominates, followed by Rayleigh ~2-3%, Photoelectric ~1%, Pair <0.1%
    probs = {
        'compton': 0.9625,
        'rayleigh': 0.0268,
        'photoelectric': 0.0102,
        'pair': 0.0005
    }
    proc_names = list(probs.keys())
    proc_p = np.array(list(probs.values()))
    proc_p /= proc_p.sum()
    
    E0 = 2.0
    mc2 = 0.510998
    
    events = []
    
    for _ in range(n):
        proc = np.random.choice(proc_names, p=proc_p)
        evt = {'process': proc, 'e_in_MeV': 2.0}
        
        if proc == 'compton':
            # Klein-Nishina differential cross section sampling
            def kn_weight(cos_t):
                alpha = E0 / mc2
                E_out = E0 / (1 + alpha * (1 - cos_t))
                return (E_out/E0)**2 * (E_out/E0 + E0/E_out - (1 - cos_t**2))
            
            max_w = kn_weight(1.0)
            while True:
                c = 2*np.random.rand()-1
                if np.random.rand() * max_w <= kn_weight(c):
                    cos_sc = c
                    break
            
            alpha = E0/mc2
            E_out = E0 / (1 + alpha*(1-cos_sc))
            t_e = E0 - E_out
            
            # Electron angle relation
            tan_theta_sc2 = np.sqrt(1-cos_sc**2) / (1+cos_sc)
            cot_theta_e = (1+alpha)*tan_theta_sc2
            cos_e = 1.0 / np.sqrt(1 + cot_theta_e**2)
            
            evt['e_out_MeV'] = round(E_out, 6)
            evt['cos_scatter'] = round(cos_sc, 6)
            evt['t_e_MeV'] = round(t_e, 6)
            evt['cos_electron'] = round(cos_e, 6)
            evt['t_positron_MeV'] = None
            
        elif proc == 'photoelectric':
            # Neglect binding energy at 2 MeV
            t_e = E0
            # Strongly forward peaked
            cos_e = np.clip(np.random.normal(0.98, 0.02), 0.8, 1.0)
            
            evt['e_out_MeV'] = None
            evt['cos_scatter'] = None
            evt['t_e_MeV'] = round(t_e, 6)
            evt['cos_electron'] = round(float(cos_e), 6)
            evt['t_positron_MeV'] = None
            
        elif proc == 'rayleigh':
            # Elastic coherent scattering
            E_out = E0
            t_e = 0.0
            # Forward peaked, sample small angles
            delta = -np.log(np.random.rand()*0.5+0.5)*0.02
            cos_sc = np.cos(delta)
            
            evt['e_out_MeV'] = round(E_out, 6)
            evt['cos_scatter'] = round(float(cos_sc), 6)
            evt['t_e_MeV'] = round(t_e, 6)
            evt['cos_electron'] = None
            evt['t_positron_MeV'] = None
            
        elif proc == 'pair':
            T_tot = E0 - 1.022
            f = np.random.uniform(0.4, 0.6) # Roughly equal sharing near threshold
            t_e = T_tot * f
            t_pos = T_tot * (1-f)
            cos_e = np.clip(np.random.normal(0.98, 0.01), 0.9, 1.0)
            
            evt['e_out_MeV'] = None
            evt['cos_scatter'] = None
            evt['t_e_MeV'] = round(t_e, 6)
            evt['cos_electron'] = round(float(cos_e), 6)
            evt['t_positron_MeV'] = round(t_pos, 6)
            
        events.append(evt)
        
    return events

events = generate_events()
method_desc = ("Events sampled probabilistically according to NIST XCOM mass attenuation coefficients for liquid water at 2.0 MeV. Compton scattering angles drawn from Klein-Nishina distribution via rejection sampling; photoelectric electrons assigned forward-peaked angles; Rayleigh scattering simulated with exponentially distributed small-angle kicks; pair production splits kinetic energy symmetrically with forward emission."[:99])

output = {"method": method_desc, "events": events}
print(json.dumps(output, indent=2))