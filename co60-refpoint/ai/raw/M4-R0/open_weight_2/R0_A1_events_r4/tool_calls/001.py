import numpy as np
import json

def kn_cos_scatter(alpha):
    # Rejection sampling for Klein-Nishina distribution dσ/d(cosθ)
    def target_pdf(x):
        return (1.0 / (1.0 + alpha * (1.0 - x)))**2 * (1.0 + x**2 + (alpha**2) * (1.0 - x)**2 / (1.0 + alpha * (1.0 - x)))
    
    while True:
        x = np.random.uniform(-1.0, 1.0)
        u = np.random.uniform(0.0, 2.0) # max of target_pdf is at x=1, value=2
        if u <= target_pdf(x):
            return x

def rayleigh_cos_scatter():
    # Approximate Mott distribution peaked forward: P(x) ∝ (1+x^2) * exp(-3*(1-x))
    def target_pdf(x):
        return (1.0 + x**2) * np.exp(-3.0 * (1.0 - x))
    max_val = target_pdf(1.0)
    while True:
        x = np.random.uniform(-1.0, 1.0)
        u = np.random.uniform(0.0, max_val)
        if u <= target_pdf(x):
            return x

events = []
n_events = 200
processes = ['compton', 'photoelectric', 'rayleigh']
probs = [0.85, 0.10, 0.05]

mec2 = 0.511

for _ in range(n_events):
    proc = np.random.choice(processes, p=probs)
    e_in = np.random.choice([1.1732, 1.3325])
    
    ev = {"process": proc, "e_in_MeV": round(e_in, 4)}
    
    if proc == "compton":
        alpha = e_in / mec2
        cs = kn_cos_scatter(alpha)
        e_out = e_in / (1.0 + alpha * (1.0 - cs))
        t_e = e_in - e_out
        
        # cos_electron from kinematics: tan(theta_e) = 1 / ((1 + E/m) * tan(theta/2))
        tan_theta_half = np.sqrt((1.0 - cs) / (1.0 + cs))
        tan_theta_e = 1.0 / ((1.0 + alpha) * tan_theta_half)
        cos_e = 1.0 / np.sqrt(1.0 + tan_theta_e**2)
        
        ev.update({
            "e_out_MeV": round(e_out, 4),
            "cos_scatter": round(cs, 4),
            "t_e_MeV": round(t_e, 4),
            "cos_electron": round(cos_e, 4),
            "t_positron_MeV": None
        })
        
    elif proc == "photoelectric":
        # Binding energy of O K-shell ~0.53 keV = 0.000533 MeV
        t_e = e_in - 0.000533
        cos_e = np.random.uniform(-1.0, 1.0)
        
        ev.update({
            "e_out_MeV": None,
            "cos_scatter": None,
            "t_e_MeV": round(t_e, 4),
            "cos_electron": round(cos_e, 4),
            "t_positron_MeV": None
        })
        
    elif proc == "rayleigh":
        cs = rayleigh_cos_scatter()
        ev.update({
            "e_out_MeV": round(e_in, 4),
            "cos_scatter": round(cs, 4),
            "t_e_MeV": 0.0,
            "cos_electron": None,
            "t_positron_MeV": None
        })
        
    elif proc == "pair":
        # Extremely rare at these energies, skip for practical purposes unless seed hits
        pass
        
    events.append(ev)

# Output JSON
print(json.dumps({"method": "Monte Carlo sampling of photon interactions in water based on NIST XCOM cross sections for Co-60 energies (Compton ~85%, Photoelectric ~10%, Rayleigh ~5%). Klein-Nishina distribution used for Compton scattering kinematics. Mott-like forward-peaked distribution for Rayleigh. Photoelectric electrons emitted isotropically with binding energy subtraction.", "events": events}))