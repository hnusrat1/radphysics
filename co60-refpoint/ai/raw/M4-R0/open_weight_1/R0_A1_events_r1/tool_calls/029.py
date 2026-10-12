# Main simulation loop
import time

def run_simulation(max_photons=600000):
    events = []
    n_histories = 0
    t0 = time.time()
    
    for i in range(max_photons):
        n_histories += 1
        
        # Source photon
        Ein = source_E()
        di = source_dir()
        
        # Entry at phantom surface (z=0)
        t_surf = -src[2] / di[2]
        pos = src + t_surf * di
        
        # Track through phantom
        mu = mu_t(Ein)
        active = True
        
        while active:
            d_int = -np.log(1.0 - rng.random()) / mu
            d_ext = dist_exit(pos, di)
            
            if d_int > d_ext:
                break
            
            pos_int = pos + d_int * di
            
            if in_scoring(pos_int):
                proc = choose_proc(Ein)
                
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
                    if E_out > 0.01:
                        di = scatter_dir(di, ct)
                        pos = pos_int
                        Ein = E_out
                        mu = mu_t(Ein)
                    else:
                        active = False
                        
                elif proc == 'photoelectric':
                    events.append({
                        'process': 'photoelectric',
                        'e_in_MeV': Ein,
                        'e_out_MeV': None,
                        'cos_scatter': None,
                        't_e_MeV': Ein,
                        'cos_electron': rng.uniform(0.9, 1.0),
                        't_positron_MeV': None
                    })
                    active = False
                    
                elif proc == 'pair':
                    _, _, te, ce, tp, _ = pair_proc(Ein)
                    events.append({
                        'process': 'pair',
                        'e_in_MeV': Ein,
                        'e_out_MeV': None,
                        'cos_scatter': None,
                        't_e_MeV': te,
                        'cos_electron': ce,
                        't_positron_MeV': tp
                    })
                    active = False
                    
                elif proc == 'rayleigh':
                    E_out, ct, _, _ = ray_scat(Ein)
                    events.append({
                        'process': 'rayleigh',
                        'e_in_MeV': Ein,
                        'e_out_MeV': E_out,
                        'cos_scatter': ct,
                        't_e_MeV': 0.0,
                        'cos_electron': None,
                        't_positron_MeV': None
                    })
                    if E_out > 0.01:
                        di = scatter_dir(di, ct)
                        pos = pos_int
                        # Rayleigh: energy unchanged
                        mu = mu_t(Ein)
                    else:
                        active = False
            else:
                # Interaction not in scoring volume
                # Process scatter or absorb for tracking
                proc = choose_proc(Ein)
                if proc == 'compton':
                    E_out, ct, _, _ = comp_scat(Ein)
                    if E_out > 0.01:
                        di = scatter_dir(di, ct)
                        pos = pos_int
                        Ein = E_out
                        mu = mu_t(Ein)
                    else:
                        active = False
                elif proc == 'rayleigh':
                    E_out, ct, _, _ = ray_scat(Ein)
                    if E_out > 0.01:
                        di = scatter_dir(di, ct)
                        pos = pos_int
                        mu = mu_t(Ein)
                    else:
                        active = False
                else:
                    # photoelectric or pair: absorbed
                    active = False
        
        if len(events) >= 200:
            break
            
        if n_histories % 100000 == 0:
            print(f"  {n_histories} histories, {len(events)} events, {time.time()-t0:.1f}s")
    
    print(f"Done: {n_histories} histories, {len(events)} events in {time.time()-t0:.1f}s")
    return events

# Run with 600k photons
print("Starting simulation...")
events = run_simulation(600000)
print(f"Got {len(events)} events")