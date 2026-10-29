import sys, json
import numpy as np

# The setup from previous calls persists

# ============ PRIMARY RATE ============
N_MC = 500000
prim_tot = 0.0
prim_by = {'compton':0.0, 'photoelectric':0.0, 'pair':0.0, 'rayleigh':0.0}

for _ in range(N_MC):
    E, d, x, y = src_samp()
    dz = d[2]
    if dz <= 0: continue
    
    # Ray position at z=zlo (scoring volume entry)
    x_ep = x * (SSD + zlo) / SSD
    y_ep = y * (SSD + zlo) / SSD
    if np.sqrt(x_ep**2 + y_ep**2) > Rc: continue
    
    # Path through scoring volume (along z)
    path = 0.2 / dz
    
    muE = mu_t(E)
    # Attenuation path = distance in water from surface to scoring entry
    # This is the slant path through water
    t_sc = (zlo - zs) / dz
    att = np.exp(-muE * t_sc)
    p_int = 1.0 - np.exp(-muE * path)
    w = att * p_int
    
    prim_tot += w
    mt = mu_c(E) + mu_ph(E) + mu_pp(E) + mu_rl(E)
    prim_by['compton'] += w * mu_c(E) / mt
    prim_by['photoelectric'] += w * mu_ph(E) / mt
    prim_by['pair'] += w * mu_pp(E) / mt
    prim_by['rayleigh'] += w * mu_rl(E) / mt

prim_tot /= N_MC
for k in prim_by: prim_by[k] /= N_MC

print("Primary rate: {:.8f}".format(prim_tot), file=sys.stderr)
for k,v in prim_by.items():
    print("  {}: {:.10f}".format(k,v), file=sys.stderr)