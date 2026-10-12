import numpy as np
import time

rng = np.random.default_rng(42)

# --- geometry parameters ---
# Source at origin, beam along +z
# Phantom: z in [100,130], |x|<15, |y|<15
# Scoring cylinder: radius 1, z in [109.9,110.1], centered on axis

def inside_cube(x,y,z):
    return (np.abs(x)<15)&(np.abs(y)<15)&(z>100)&(z<130)

def in_scoring(x,y,z):
    return (x*x+y*y<1.0)&(z>109.9)&(z<110.1)

# --- cross sections (vectorized) ---
def get_mus(E):
    """return (compton, photo, pair, rayleigh) mu in cm^-1 for water, arrays"""
    # compton via interpolation on KN grid
    s = np.interp(np.log(np.clip(E,1e-3,10)), np.log(Egrid), sig_grid, left=sig_grid[0], right=sig_grid[-1])
    comp = Ne * s
    # photo
    logE = np.log(E)
    x = np.clip(E, E_ph[0], E_ph[-1])
    ph = np.exp(np.interp(logE, np.log(E_ph), np.log(mu_ph)))
    # pair
    pa = np.where(E>1.022, np.exp(np.interp(logE, np.log(E_pa), np.log(mu_pa))), 0.0)
    # coherent
    co = np.exp(np.interp(logE, np.log(E_co), np.log(mu_co)))
    return comp, ph, pa, co

# --- KN sampler for Compton ---
def sample_compton(E, rng):
    """E: array; returns mu_s, Ep, cos_el"""
    n = len(E)
    mu = np.empty(n)
    alive = np.ones(n, bool)
    while alive.any():
        mu_alive = rng.uniform(-1,1,n)[alive]
        E_alive = E[alive]
        f = kn_f(mu_alive, E_alive)
        mask = rng.uniform(0,2,len(mu_alive)) < f
        idx = np.where(alive)[0][mask]
        mu[idx] = mu_alive[mask]
        alive[idx] = False
    Ep = E / (1.0 + (E/me)*(1.0-mu))
    # electron KE
    te = E - Ep
    # cos_electron
    num = E - Ep*mu
    den = np.sqrt(E*E + Ep*Ep - 2.0*E*Ep*mu)
    cos_el = np.where(den>1e-15, num/den, 0.0)
    return mu, Ep, te, cos_el

# --- transport for one batch ---
def simulate_batch(n_primary, rng):
    events = []  # dict of lists
    # start primary photons at phantom entry
    # sample incident direction uniformly in solid angle for 10x10 field at z=100
    # sample entry point on z=100 plane within the 10x10 square
    x0 = rng.uniform(-5,5,n_primary)
    y0 = rng.uniform(-5,5,n_primary)
    # direction from source (0,0,0) to entry point
    # source at (0,0,0), entry at (x0,y0,100)
    dx = x0 / np.sqrt(x0*x0 + y0*y0 + 10000.0)
    dy = y0 / np.sqrt(x0*x0 + y0*y0 + 10000.0)
    dz = 100.0 / np.sqrt(x0*x0 + y0*y0 + 10000.0)
    # for small angles dx,dy ~ x0/100, dz ~ 1
    px = x0.copy()
    py = y0.copy()
    pz = np.full(n_primary, 100.0)
    energy = np.full(n_primary, 2.0)
    ux = dx.copy(); uy = dy.copy(); uz = dz.copy()
    
    alive = np.ones(n_primary, bool)
    
    def kill_mask(mask):
        alive[mask] = False
    
    # We'll maintain list of new photons to inject (annihilation)
    extra_px = []; extra_py = []; extra_pz = []
    extra_ux = []; extra_uy = []; extra_uz = []
    extra_E  = []
    
    # loop
    for iteration in range(100):
        n = np.sum(alive)
        if n == 0 and len(extra_px)==0:
            break
        
        # gather arrays
        if n > 0:
            # indices of alive
            idx = np.where(alive)[0]
            E = energy[idx]
            c,ph,pa,co = get_mus(E)
            tot = c+ph+pa+co
            
            # sample step
            s = -np.log(rng.uniform(1e-10,1,n)) / tot
            # move
            px[idx] += s * ux[idx]
            py[idx] += s * uy[idx]
            pz[idx] += s * uz[idx]
            
            # which are inside cube
            inside = inside_cube(px[idx], py[idx], pz[idx])
            inside_idx = idx[inside]
            n_inside = len(inside_idx)
            
            if n_inside > 0:
                E_i = E[inside]
                c_i = c[inside]; ph_i=ph[inside]; pa_i=pa[inside]; co_i=co[inside]
                tot_i = tot[inside]
                
                # Check scoring volume
                in_sc = in_scoring(px[inside_idx], py[inside_idx], pz[inside_idx])
                
                # Sample process
                u = rng.uniform(0, tot_i, n_inside)
                # order: compton, photoelectric, pair, rayleigh
                # cumulatives
                c_cum = c_i
                ph_cum = c_i+ph_i
                pa_cum = c_i+ph_i+pa_i
                # totals = co_cum
                proc = np.full(n_inside, -1)
                proc[u<=c_cum] = 0  # compton
                proc[(u>c_cum)&(u<=ph_cum)] = 1  # photo
                proc[(u>ph_cum)&(u<=pa_cum)] = 2  # pair
                proc[(u>pa_cum)] = 3  # rayleigh
                
                # --- Record events ---
                for p_val in [0,1,2,3]:
                    mask_p = (proc==p_val)
                    if not mask_p.any(): continue
                    ev_idx = inside_idx[mask_p]
                    # scoring check
                    sc_idx = ev_idx[in_sc[mask_p]]
                    if len(sc_idx)>0:
                        E_in  = energy[sc_idx]
                        if p_val == 0:  # compton
                            mu_s, Ep, te, cos_el = sample_compton(E_in, rng)
                            for i in range(len(E_in)):
                                events.append({
                                    'process':'compton',
                                    'e_in_MeV': float(E_in[i]),
                                    'e_out_MeV': float(Ep[i]),
                                    'cos_scatter': float(mu_s[i]),
                                    't_e_MeV': float(te[i]),
                                    'cos_electron': float(cos_el[i]),
                                    't_positron_MeV': None
                                })
                        elif p_val == 1:  # photoelectric
                            te = E_in  # ignore binding
                            cos_el = rng.uniform(-1,1,len(E_in))
                            for i in range(len(E_in)):
                                events.append({
                                    'process':'photoelectric',
                                    'e_in_MeV': float(E_in[i]),
                                    'e_out_MeV': None,
                                    'cos_scatter': None,
                                    't_e_MeV': float(te[i]),
                                    'cos_electron': float(cos_el[i]),
                                    't_positron_MeV': None
                                })
                        elif p_val == 2:  # pair
                            t_avail = E_in - 1.022
                            # sample positron kinetic energy share uniform
                            share = rng.uniform(0,1,len(E_in))
                            t_pos = share * t_avail
                            t_e = t_avail - t_pos
                            # angles: each lepton forward peaked
                            # electron
                            E_total_e = t_e + me
                            # characteristic angle me/E_total
                            theta0_e = me / E_total_e
                            # sample from dipole-like: theta = -me/E * ln? use exponential
                            # sample theta from pdf ∝ theta/(1+(theta/θ0)²)²
                            # use transformation: theta = θ0 * tan(π u/2)   (some distribution)
                            # Actually simpler: use Rayleigh-distributed angle with scale θ0
                            # Use: theta = θ0 * sqrt(-2*log(u))
                            # But the dipole distribution is better approximated by:
                            # theta = θ0 * tan(π*u/2)  gives pdf ∝ 1/(1+(θ/θ0)²)
                            # Hmm, let me just use exponential: θ = -θ0 * log(u)
                            # For consistency, let me sample from: t = -log(u), theta = θ0 * sqrt(t)
                            u1 = rng.random(len(E_in)); theta_e = theta0_e * np.sqrt(-2.0*np.log(1e-10+u1*(1-1e-10)))
                            cos_el = np.cos(theta_e)
                            # for sampling, set cos_electron from incident direction
                            for i in range(len(E_in)):
                                events.append({
                                    'process':'pair',
                                    'e_in_MeV': float(E_in[i]),
                                    'e_out_MeV': None,
                                    'cos_scatter': None,
                                    't_e_MeV': float(t_e[i]),
                                    'cos_electron': float(cos_el[i]),
                                    't_positron_MeV': float(t_pos[i])
                                })
                            # Emit annihilation photons
                            for i in range(len(E_in)):
                                # 2 photons at 0.511 MeV, isotropic
                                for _ in range(2):
                                    u1 = rng.random(); u2 = rng.random()*2*np.pi
                                    cz = 2.0*u1-1.0; sz = np.sqrt(1.0-cz*cz)
                                    nx = sz*np.cos(u2); ny = sz*np.sin(u2); nz = cz
                                    extra_px.append(px[sc_idx[i]])
                                    extra_py.append(py[sc_idx[i]])
                                    extra_pz.append(pz[sc_idx[i]])
                                    extra_ux.append(nx); extra_uy.append(ny); extra_uz.append(nz)
                                    extra_E.append(0.511)
                        elif p_val == 3:  # rayleigh
                            # sample small scattering angle
                            # characteristic angle for coherent scattering
                            # Use: θ0 = 0.0007 / E (rad) ~ coherent scale for water
                            theta0 = np.clip(0.0007 / E_in, 1e-6, 0.5)
                            u1 = rng.random(len(E_in))
                            theta = theta0 * np.sqrt(-2.0*np.log(1e-10+u1*(1-1e-10)))
                            cos_sc = np.cos(theta)
                            for i in range(len(E_in)):
                                events.append({
                                    'process':'rayleigh',
                                    'e_in_MeV': float(E_in[i]),
                                    'e_out_MeV': float(E_in[i]),
                                    'cos_scatter': float(cos_sc[i]),
                                    't_e_MeV': 0.0,
                                    'cos_electron': None,  # null
                                    # Actually cos_electron null for rayleigh
                                })
                            # But the schema requires cos_electron field; for rayleigh it's null
                            # I need to fix the above dict -- add null for cos_electron and t_pos
                            # Let me redo rayleigh entries properly
                            # I'll replace the rayleigh entries above
                        else:
                            pass
                
                # Now process interactions and update state
                # For compton: update direction and energy
                mask_c = (proc==0)
                if mask_c.any():
                    idx_c = inside_idx[mask_c]
                    # sample compton angles for those
                    mu_s, Ep, te, cos_el = sample_compton(energy[idx_c], rng)
                    energy[idx_c] = Ep
                    # new direction from scattering
                    # incident direction d = (ux,uy,uz)
                    d = np.column_stack([ux[idx_c], uy[idx_c], uz[idx_c]])
                    # construct basis: e1_perp to d
                    # choose reference axis: z if d not near z, else x
                    ref = np.tile(np.array([0.0,0.0,1.0]), (len(idx_c),1))
                    near_z = np.abs(d[:,2]) > 0.9
                    ref[near_z] = [1.0,0.0,0.0]
                    e1 = np.cross(d, ref)
                    nrm1 = np.linalg.norm(e1, axis=1)
                    e1 = e1 / nrm1[:, np.newaxis]
                    e2 = np.cross(d, e1)
                    # azimuth
                    phi = rng.uniform(0,2*np.pi, len(idx_c))
                    st = np.sqrt(1.0 - mu_s*mu_s)
                    newd = (st*np.cos(phi))[:,np.newaxis]*e1 + (st*np.sin(phi))[:,np.newaxis]*e2 + mu_s[:,np.newaxis]*d
                    ux[idx_c] = newd[:,0]; uy[idx_c] = newd[:,1]; uz[idx_c] = newd[:,2]
                    # keep alive
                
                # For photoelectric: kill
                mask_pe = (proc==1)
                if mask_pe.any():
                    kill_mask(inside_idx[mask_pe])
                
                # For pair: kill
                mask_pa = (proc==2)
                if mask_pa.any():
                    kill_mask(inside_idx[mask_pa])
                    # annihilation photons already added via lists
                
                # For rayleigh: update direction, energy same
                mask_r = (proc==3)
                if mask_r.any():
                    idx_r = inside_idx[mask_r]
                    theta0 = np.clip(0.0007 / energy[idx_r], 1e-6, 0.5)
                    u1 = rng.random(len(idx_r))
                    theta = theta0 * np.sqrt(-2.0*np.log(1e-10+u1*(1-1e-10)))
                    mu_r = np.cos(theta)
                    d = np.column_stack([ux[idx_r], uy[idx_r], uz[idx_r]])
                    ref = np.tile(np.array([0.0,0.0,1.0]), (len(idx_r),1))
                    near_z = np.abs(d[:,2]) > 0.9
                    ref[near_z] = [1.0,0.0,0.0]
                    e1 = np.cross(d, ref)
                    nrm1 = np.linalg.norm(e1, axis=1)
                    e1 = e1 / nrm1[:, np.newaxis]
                    e2 = np.cross(d, e1)
                    phi = rng.uniform(0,2*np.pi, len(idx_r))
                    st = np.sqrt(1.0 - mu_r*mu_r)
                    newd = (st*np.cos(phi))[:,np.newaxis]*e1 + (st*np.sin(phi))[:,np.newaxis]*e2 + mu_r[:,np.newaxis]*d
                    ux[idx_r] = newd[:,0]; uy[idx_r] = newd[:,1]; uz[idx_r] = newd[:,2]
                    # keep alive, energy unchanged
                
            # Those outside cube are killed
            outside_idx = idx[~inside]
            kill_mask(outside_idx)
        
        # Add annihilation photons from extra list
        if len(extra_px) > 0:
            # Append to main arrays
            n_add = len(extra_px)
            pxt = np.array(extra_px + [])
            pyt = np.array(extra_py + [])
            pzt = np.array(extra_pz + [])
            uxt = np.array(extra_ux + [])
            uyt = np.array(extra_uy + [])
            uzt = np.array(extra_uz + [])
            Et  = np.array(extra_E + [])
            # Actually need to extend arrays... this is getting messy.
            # Let me restructure: use dynamic list of active photons.
            # For now, just add to arrays by resizing.
            # This is messy. I'll restructure.
            pass
        
        if n == 0 and len(extra_px)==0:
            break
    
    return events

# test small
ev = simulate_batch(2000, rng)
print("Events:", len(ev))