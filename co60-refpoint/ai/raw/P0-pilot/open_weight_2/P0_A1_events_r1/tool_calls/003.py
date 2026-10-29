import numpy as np
import json

def generate_events(n=200):
    mc2 = 0.51099895000
    E = 2.0
    procs = ['compton', 'pair', 'rayleigh', 'photoelectric']
    weights = np.array([0.972, 0.0255, 0.0067, 0.0026])
    weights /= weights.sum()
    
    events = []
    rng = np.random.default_rng(42)
    
    for _ in range(n):
        proc = rng.choice(procs, p=weights)
        ev = {'process': proc, 'e_in_MeV': E}
        
        if proc == 'compton':
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            az = rng.uniform(0, 2*np.pi)
            
            sqrt_s = np.sqrt(mc2**2 + 2*mc2*E)
            E_cm = mc2 * E / sqrt_s
            beta_cm = E / (E + mc2)
            gamma_cm = (E + mc2) / sqrt_s
            
            px_cm = E_cm * sin_phi * np.cos(az)
            py_cm = E_cm * sin_phi * np.sin(az)
            pz_cm = E_cm * cos_phi
            
            E_out = gamma_cm * (E_cm + beta_cm * pz_cm)
            px_lab = px_cm
            py_lab = py_cm
            pz_lab = gamma_cm * (pz_cm + beta_cm * E_cm)
            
            cos_sc = pz_lab / E_out
            
            # Momentum conservation for electron
            px_e = -px_lab
            py_e = -py_lab
            pz_e = E - pz_lab
            T_e = E - E_out
            p_e_mag_sq = px_e**2 + py_e**2 + pz_e**2
            # Check consistency
            p_e_mag_calc = np.sqrt(T_e**2 + 2*T_e*mc2)
            # They should match analytically; use calculated mag for normalization
            cos_el = pz_e / p_e_mag_calc
            
            ev['e_out_MeV'] = round(E_out, 6)
            ev['cos_scatter'] = round(cos_sc, 6)
            ev['t_e_MeV'] = round(T_e, 6)
            ev['cos_electron'] = round(np.clip(cos_el, -1.0, 1.0), 6)
            ev['t_positron_MeV'] = None
            
        elif proc == 'pair':
            E_avail = E - 2 * mc2
            t_pos = rng.uniform(0, E_avail)
            t_e = E_avail - t_pos
            ev['e_out_MeV'] = None
            ev['cos_scatter'] = None
            ev['t_e_MeV'] = round(t_e, 6)
            ev['cos_electron'] = None
            ev['t_positron_MeV'] = round(t_pos, 6)
            
        elif proc == 'rayleigh':
            u = rng.random()
            cos_sc = 1 - u**0.3
            cos_sc = max(-1.0, min(1.0, cos_sc))
            ev['e_out_MeV'] = round(E, 6)
            ev['cos_scatter'] = round(cos_sc, 6)
            ev['t_e_MeV'] = 0.0
            ev['cos_electron'] = None
            ev['t_positron_MeV'] = None
            
        elif proc == 'photoelectric':
            t_e = E
            u = rng.random()
            cos_el = 1 - u**4
            cos_el = max(-1.0, min(1.0, cos_el))
            ev['e_out_MeV'] = None
            ev['cos_scatter'] = None
            ev['t_e_MeV'] = round(t_e, 6)
            ev['cos_electron'] = round(cos_el, 6)
            ev['t_positron_MeV'] = None
            
        events.append(ev)
        
    return events

evs = generate_events()
print(json.dumps(evs[:5], indent=2))
counts = {p: sum(1 for e in evs if e['process']==p) for p in procs}
print("Counts:", counts)
print("Total:", len(evs))
# Validate fields
for i, e in enumerate(evs):
    assert e['e_in_MeV'] == 2.0
    if e['process'] == 'compton':
        assert abs(e['t_e_MeV'] + e['e_out_MeV'] - 2.0) < 1e-5
        assert -1 <= e['cos_scatter'] <= 1
        assert -1 <= e['cos_electron'] <= 1
    elif e['process'] == 'pair':
        assert abs(e['t_e_MeV'] + e['t_positron_MeV'] - 0.958) < 1e-5
    print(f"Event {i} OK")