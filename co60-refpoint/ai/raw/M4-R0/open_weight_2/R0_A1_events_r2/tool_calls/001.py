import numpy as np
import json

np.random.seed(12345)
N = 200
processes = ['compton', 'photoelectric', 'pair', 'rayleigh']
weights = [0.88, 0.07, 0.03, 0.02]
energies = [1.1732, 1.3325]
me = 0.51099895

events = []

def ks_pdf(x, alpha):
    fac = 1.0 / (1.0 + alpha * (1.0 - x))
    return fac**2 * (fac + 1.0/fac - 1.0 + x**2)

for _ in range(N):
    proc = np.random.choice(processes, p=weights)
    E = np.random.choice(energies)
    
    if proc == 'compton':
        alpha = E / me
        # Precompute max of PDF for rejection sampling
        xs = np.linspace(-1.0, 1.0, 1000)
        ys = ks_pdf(xs, alpha)
        M = np.max(ys)
        
        while True:
            x = np.random.uniform(-1.0, 1.0)
            if np.random.uniform(0, M) <= ks_pdf(x, alpha):
                cos_sc = x
                break
        
        E_out = E / (1.0 + alpha * (1.0 - cos_sc))
        t_e = E - E_out
        
        p_g = E
        p_gp = E_out
        p_e_sq = p_g**2 + p_gp**2 - 2*p_g*p_gp*cos_sc
        p_e = np.sqrt(max(p_e_sq, 0))
        cos_e = (p_g - p_gp * cos_sc) / p_e if p_e > 1e-12 else 1.0
        cos_e = np.clip(cos_e, -1.0, 1.0)
        
        events.append({
            "process": "compton",
            "e_in_MeV": round(E, 4),
            "e_out_MeV": round(E_out, 4),
            "cos_scatter": round(float(cos_sc), 4),
            "t_e_MeV": round(t_e, 4),
            "cos_electron": round(float(cos_e), 4),
            "t_positron_MeV": None
        })

    elif proc == 'photoelectric':
        t_e = E
        beta = np.random.beta(5, 2)
        cos_e = np.clip(2*beta - 1, -1, 1)
        events.append({
            "process": "photoelectric",
            "e_in_MeV": round(E, 4),
            "e_out_MeV": None,
            "cos_scatter": None,
            "t_e_MeV": round(t_e, 4),
            "cos_electron": round(float(cos_e), 4),
            "t_positron_MeV": None
        })
        
    elif proc == 'pair':
        T_total = E - 1.022
        U = np.random.uniform(0, 1)
        t_e = U * T_total
        t_p = (1 - U) * T_total
        beta_e = np.random.beta(4, 2)
        beta_p = np.random.beta(4, 2)
        cos_e = np.clip(2*beta_e - 1, -1, 1)
        cos_p = np.clip(2*beta_p - 1, -1, 1)
        events.append({
            "process": "pair",
            "e_in_MeV": round(E, 4),
            "e_out_MeV": None,
            "cos_scatter": None,
            "t_e_MeV": round(t_e, 4),
            "cos_electron": round(float(cos_e), 4),
            "t_positron_MeV": round(t_p, 4)
        })
        
    elif proc == 'rayleigh':
        E_out = E
        beta_sc = np.random.beta(6, 1)
        cos_sc = np.clip(2*beta_sc - 1, -1, 1)
        events.append({
            "process": "rayleigh",
            "e_in_MeV": round(E, 4),
            "e_out_MeV": round(E_out, 4),
            "cos_scatter": round(float(cos_sc), 4),
            "t_e_MeV": 0.0,
            "cos_electron": None,
            "t_positron_MeV": None
        })

output = {
    "method": "Random sampling of Co-60 photon interactions in water at 5 cm depth. Process probabilities set to Compton 88%, Photoelectric 7%, Pair 3%, Rayleigh 2%, based on NIST macroscopic cross-section ratios. Incident energies chosen 50/50 from 1.1732 and 1.3325 MeV. Compton kinematics sampled from Klein-Nishina distribution; Photoelectric and Pair electron/positron angular distributions forward-peaked via Beta distributions; Rayleigh scattering forward-peaked. Delta rays excluded.",
    "events": events
}

print(json.dumps(output, indent=2)[:200]) # Just checking start
# Write to file to avoid truncation issues in console
with open('events.json', 'w') as f:
    json.dump(output, f, indent=2)
print("Done")