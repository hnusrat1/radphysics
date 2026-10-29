import numpy as np
import json

def generate_events():
    mc2 = 0.51099895000
    E = 2.0
    procs = ['compton', 'pair', 'rayleigh', 'photoelectric']
    weights = np.array([0.972, 0.0255, 0.0067, 0.0026])
    weights /= weights.sum()
    
    events = []
    rng = np.random.default_rng(123)
    
    for _ in range(200):
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
            pz_lab = gamma_cm * (pz_cm + beta_cm * E_cm)
            cos_sc = pz_lab / E_out
            
            pz_e = E - pz_lab
            px_e = -px_cm
            py_e = -py_cm
            
            p_e_mag_sq = px_e**2 + py_e**2 + pz_e**2
            p_e_mag = np.sqrt(p_e_mag_sq)
            cos_el = pz_e / p_e_mag
            
            T_e = E - E_out
            
            ev['e_out_MeV'] = round(E_out, 6)
            ev['cos_scatter'] = round(cos_sc, 6)
            ev['t_e_MeV'] = round(T_e, 6)
            ev['cos_electron'] = round(np.clip(cos_el, -1.0, 1.0), 6)
            ev['t_positron_MeV'] = None
            
        elif proc == 'pair':
            E_avail = E - 2 * mc2
            t_pos = rng.uniform(0, E_avail)
            t_e = E_avail - t_pos
            
            u = rng.random()
            cos_el = 1 - u**2
            u_p = rng.random()
            cos_pos = 1 - u_p**2
            
            ev['e_out_MeV'] = None
            ev['cos_scatter'] = None
            ev['t_e_MeV'] = round(t_e, 6)
            ev['cos_electron'] = round(np.clip(cos_el, 0.0, 1.0), 6)
            ev['t_positron_MeV'] = round(t_pos, 6)
            
        elif proc == 'rayleigh':
            u = rng.random()
            cos_sc = 1 - u**0.2
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
out = {
    "method": "Sampled interaction processes proportional to mass attenuation coefficients for 2 MeV photons in water (~97% Compton, ~2.6% pair, ~0.7% Rayleigh, ~0.3% photoelectric). Compton events used exact relativistic kinematics via CM-to-Lab Lorentz boosts ensuring four-momentum conservation. Pair production energies distributed uniformly above threshold. Angular distributions for all non-Rayleigh processes were biased forward-peaked consistent with high-energy cross-sections.",
    "events": evs
}
with open('events.json', 'w') as f:
    json.dump(out, f)
print("Written to events.json")