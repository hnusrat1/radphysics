import numpy as np
import json

def generate_events(n=200):
    mc2 = 0.51099895000
    E = 2.0
    alpha = E / mc2
    
    # Relative cross sections for 2 MeV in water
    # Compton ~ 0.972, Pair ~ 0.0255, Rayleigh ~ 0.0067, Photoelec ~ 0.0026
    procs = ['compton', 'pair', 'rayleigh', 'photoelectric']
    weights = np.array([0.972, 0.0255, 0.0067, 0.0026])
    weights /= weights.sum()
    
    events = []
    rng = np.random.default_rng(42) # Fixed seed for reproducibility, though random sample is requested
    
    for _ in range(n):
        proc = rng.choice(procs, p=weights)
        ev = {'process': proc, 'e_in_MeV': E}
        
        if proc == 'compton':
            # Sample CM scattering angle cos_phi uniform in [-1, 1]
            cos_phi = rng.uniform(-1, 1)
            # Lab scattering angle cos_theta
            cot_theta = (1 + alpha) * np.sqrt((1 - cos_phi) / (1 + cos_phi))
            cos_sc = 1 / np.sqrt(1 + cot_theta**2)
            
            # Energy transfer
            E_out = E / (1 + alpha * (1 - cos_phi))
            t_e = E - E_out
            
            # Electron angle
            # Momentum conservation in lab: p_e = k - k'
            # k = (0,0,E)
            # k' components:
            sin_phi = np.sqrt(1 - cos_phi**2)
            az = rng.uniform(0, 2*np.pi) # azimuthal symmetry
            k_prime_z = cos_sc * E_out
            k_prime_T = np.sqrt(E_out**2 - k_prime_z**2) # transverse momentum magnitude
            
            # electron transverse momentum equals photon's, opposite direction
            # electron longitudinal momentum
            p_e_z = E - k_prime_z
            p_e_mag_sq = p_e_z**2 + k_prime_T**2
            cos_el = p_e_z / np.sqrt(p_e_mag_sq)
            
            ev['e_out_MeV'] = round(E_out, 6)
            ev['cos_scatter'] = round(cos_sc, 6)
            ev['t_e_MeV'] = round(t_e, 6)
            ev['cos_electron'] = round(cos_el, 6)
            ev['t_positron_MeV'] = None
            
        elif proc == 'pair':
            E_avail = E - 2 * mc2
            t_pos = rng.uniform(0, E_avail)
            t_e = E_avail - t_pos
            
            ev['e_out_MeV'] = None
            ev['cos_scatter'] = None
            ev['t_e_MeV'] = round(t_e, 6)
            ev['cos_electron'] = None # Not defined for pair in prompt context usually
            ev['t_positron_MeV'] = round(t_pos, 6)
            
        elif proc == 'rayleigh':
            # Elastic scattering, highly forward peaked at 2 MeV
            # Approximate distribution using beta or exponential bias towards 1
            u = rng.random()
            cos_sc = 1 - u**0.5 # biased forward
            cos_sc = max(-1.0, min(1.0, cos_sc))
            
            ev['e_out_MeV'] = round(E, 6)
            ev['cos_scatter'] = round(cos_sc, 6)
            ev['t_e_MeV'] = 0.0
            ev['cos_electron'] = None
            ev['t_positron_MeV'] = None
            
        elif proc == 'photoelectric':
            # Absorption, electron gets almost all energy
            t_e = E
            # Forward peaked distribution ~ sin^2(theta)/(1-beta*cos)^4
            # beta ~ 0.94. Sample using rejection or simple approximation.
            # Simple: cos_e = 1 - u^3
            u = rng.random()
            cos_el = 1 - u**3
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
print(f"Total events: {len(evs)}")
print("Processes:", {p: sum(1 for e in evs if e['process']==p) for p in ['compton','pair','rayleigh','photoelectric']})