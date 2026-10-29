import numpy as np
import json

def generate_events(n=200):
    mc2 = 0.51099895000
    E = 2.0
    alpha = E / mc2
    
    procs = ['compton', 'pair', 'rayleigh', 'photoelectric']
    weights = np.array([0.972, 0.0255, 0.0067, 0.0026])
    weights /= weights.sum()
    
    events = []
    rng = np.random.default_rng(42)
    
    for _ in range(n):
        proc = rng.choice(procs, p=weights)
        ev = {'process': proc, 'e_in_MeV': E}
        
        if proc == 'compton':
            # Sample isotropic in CM
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            az = rng.uniform(0, 2*np.pi)
            
            # CM kinematics
            sqrt_s = np.sqrt(mc2**2 + 2*mc2*E)
            E_cm = mc2 * E / sqrt_s
            beta_cm = E / (E + mc2)
            gamma_cm = (E + mc2) / sqrt_s
            
            # Boost back to Lab
            # Photon direction in CM: (sin_phi*cos(az), sin_phi*sin(az), cos_phi)
            px_cm = E_cm * sin_phi * np.cos(az)
            py_cm = E_cm * sin_phi * np.sin(az)
            pz_cm = E_cm * cos_phi
            
            # Lorentz boost along Z
            E_out = gamma_cm * (E_cm + beta_cm * pz_cm)
            px_lab = px_cm
            py_lab = py_cm
            pz_lab = gamma_cm * (pz_cm + beta_cm * E_cm)
            
            cos_sc = pz_lab / E_out
            cos_az = np.cos(az) # azimuthal for electron doesn't matter for polar angle
            
            # Electron momentum in CM: opposite to photon
            px_e_cm = -px_cm
            py_e_cm = -py_cm
            # Energy in CM: sqrt(p^2 + m^2)
            E_e_cm = np.sqrt(px_e_cm**2 + py_e_cm**2 + mc2**2)
            pz_e_cm = -pz_cm
            
            # Boost electron
            T_e_lab = gamma_cm * (E_e_cm + beta_cm * pz_e_cm) - mc2
            px_e_lab = px_e_cm
            py_e_lab = py_e_cm
            pz_e_lab = gamma_cm * (pz_e_cm + beta_cm * E_e_cm)
            
            p_e_mag = np.sqrt(T_e_lab**2 + 2*T_e_lab*mc2)
            cos_el = pz_e_lab / p_e_mag
            
            ev['e_out_MeV'] = round(E_out, 6)
            ev['cos_scatter'] = round(cos_sc, 6)
            ev['t_e_MeV'] = round(T_e_lab, 6)
            ev['cos_electron'] = round(cos_el, 6)
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
            cos_sc = 1 - u**0.3 # Strongly forward peaked
            cos_sc = max(-1.0, min(1.0, cos_sc))
            ev['e_out_MeV'] = round(E, 6)
            ev['cos_scatter'] = round(cos_sc, 6)
            ev['t_e_MeV'] = 0.0
            ev['cos_electron'] = None
            ev['t_positron_MeV'] = None
            
        elif proc == 'photoelectric':
            t_e = E
            # Forward peaked
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
print(json.dumps(evs[:3], indent=2))
counts = {p: sum(1 for e in evs if e['process']==p) for p in procs}
print("Counts:", counts)
print("Total:", len(evs))