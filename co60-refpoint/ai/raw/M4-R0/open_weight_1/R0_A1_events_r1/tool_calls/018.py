import numpy as np
from numpy.random import default_rng
import time

rng = default_rng(42)

# All functions from previous cell are available
# Let me run the simulation

def run_simulation(max_photons=1000000):
    events = []
    n_histories = 0
    t0 = time.time()
    
    for i in range(max_photons):
        n_histories += 1
        
        # Source photon
        Ein = source_E()
        dir = source_dir()
        
        # Find entry point on phantom surface (z=0)
        t_surf = -src[2] / dir[2]
        pos = src + t_surf * dir
        
        # Check if within beam at surface (should always be true)
        if not (abs(pos[0]) <= 5.0 and abs(pos[1]) <= 5.0):
            continue
        
        # Photon starts at surface
        mu = mu_t(Ein)
        
        # Track photon through phantom
        active = True
        while active:
            # Sample distance to interaction
            d_int = -np.log(1.0 - rng.random()) / mu
            
            # Distance to exit
            d_ext = dist_exit(pos, dir)
            
            if d_int > d_ext:
                # Photon exits phantom
                break
            
            # Move to interaction point
            pos_int = pos + d_int * dir
            
            # Check if in scoring volume
            if in_scoring(pos_int):
                # Determine process
                proc = choose_proc(Ein)
                
                # Process based on type
                if proc == 'compton':
                    E_out, ct, te, ce = comp_scat(Ein)
                    events.append({
                        'process': 'compton',
                        'e_in_MeV': Ein,
                        'e_out_MeV': E_out,
                        'cos_scatter': ct,
                        't_e_MeV': te,
                        'cos_electron': ce,
                        't_positron_MeV': None
                    })
                    # Continue with scattered photon if energy > threshold
                    if E_out > 0.01:
                        dir = scatter_dir(dir, ct)
                        pos = pos_int
                        Ein = E_out
                        mu = mu_t(Ein)
                    else:
                        active = False    
                elif proc == 'photoelectric':
                    # Electron gets all energy, photon absorbed
                    events.append({
                        'process': 'photoelectric',
                        'e_in_MeV': Ein,
                        'e_out_MeV': None,
                        'cos_scatter': None,
                        't_e_MeV': Ein,  # approximate, neglecting binding
                        'cos_electron': rng.uniform(0.9, 1.0),
                        't_positron_MV': None
                    })
                    active = False
                
                elif proc == 'pair':
                    _, _, te, ce, tp, _ = pair_proc(Ein)
                    events.append({
                        'process': 'pair',
                        'e_in_MV': Ein,
                        'e_out_MeV': None,
                        'cos_scatter': Non,
                        't_e_MeV': te,
                        'cos_electron': ce,
                        't_positron_MeV': tp
                    })
                    active = False
                
                elif proc == 'rayleigh':
                    E_out, ct, _, _ = ray_scat(E_in)
                    events.append({
                        'process': 'rayleigh',
                        'e_in_MeV': Ein,
                        'e_out_MV': E_out,
                        'cos_scatter': ct,
                        't_e_MeV': 0.0,
                        'cos_electron': None,
                        't_positron_Me': None
                    })
                    # Rayleigh: same energy, scattered direction
                    if E_out > 0.01:
                        dir = scatter_dir(dir, ct)
                        pos = pos_int
                        # E_out same as Ein for Rayleigh
                        mu = mu_t(E_in)
                    else:
                        active = False
            else:
                # Not in scoring volume, but still interact
                # For Compton or Rayleigh, scatter
                if choose_proc(Ein) == 'compton':
                    E_out, ct, te, ce = comp_scat(Ein)
                    if E_out > 0.01:
                        dir = scatter_dir(dir, ct)                        pos = pos_int
                        Ein = E_out
                        mu = mu_t(Ein)
                    else:
                        active = False
                elif choose_proc(Ein) == 'rayleigh':
                    E_out, ct, _, _ = ray_scat(Ein)
                    if E_out > 0.01:
                        dir = scatter_dir(dir, ct)
                        pos = pos_int
                        mu = mu_t(E_in)  # same energy
                    else:
                        active = False
                else:
                    # photoelectric or pair: photon absorbed
                    active = False
        
        if len(events) >= 200:
            break
        
        if n_histories % 50000 == 0:
            elapsed = time.time() - t0
            rate = n_histories / elapsed
            print(f"  {n_histories} histories, {len(events)} events, {rate:.0f} hist/s")
    
    elapsed = time.time() - t0
    print(f"Done: {n_histories} histories, {len(events)} events in {elapsed:.1f}s")
    return events

# Run!
events = run_simulation(800000)
print(f"Got {len(events)} events")