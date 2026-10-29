import time, math, numpy as np
from collections import Counter

rng = np.random.default_rng(12345)

# Need to re-run setup code again since tool runs separate sessions
me = 0.51099895
re = 2.8179403262e-13
Ne = 3.343e23

def kn_f(mu, E):
    a = E/me
    r = 1.0/(1.0+a*(1.0-mu))
    return r*r*(1.0/r + r - (1.0-mu*mu))

def kn_sigma(E):
    mus = np.linspace(-1,1,3001)
    return np.pi*re**2*np.trapezoid(kn_f(mus,E),mus)

Egrid = np.logspace(np.log10(0.001), np.log10(10.0), 300)
sig_grid = np.array([kn_sigma(E) for E in Egrid])

E_ph = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.15,0.20,0.30,0.50,0.80,1.0,1.5,2.0])
mu_ph= np.array([4.90,0.506,0.162,0.0712,0.0375,0.0223,0.00988,0.00547,0.00240,0.00128,0.000526,0.000151,5.0e-5,3.0e-5,1.0e-5,4.7e-6])
E_pa = np.array([1.022,1.25,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0,10.0])
mu_pa= np.array([0.0,3.1e-5,1.32e-4,4.1e-4,8.1e-4,1.39e-3,2.69e-3,4.32e-3,6.19e-3,1.05e-2,1.55e-2])
E_co = np.array([0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20,0.30,0.50,0.80,1.0,1.5,2.0])
mu_co= np.array([0.30,0.060,0.022,0.0098,0.0046,0.0034,0.0017,0.0011,0.0006,0.0003,0.00015,0.00012,8e-5,4.5e-5])

def mu_comp_v(E):
    logE = np.log(np.clip(E, 1e-3, 10))
    return Ne * np.interp(logE, np.log(Egrid), sig_grid)
def mu_photo_v(E):
    logE = np.log(np.clip(E, E_ph[0], E_ph[-1]))
    return np.exp(np.interp(logE, np.log(E_ph), np.log(mu_ph)))
def mu_pair_v(E):
    mask = E > 1.022
    res = np.zeros_like(E)
    if mask.any():
        logE = np.log(np.clip(E[mask], E_pa[0], E_pa[-1]))
        res[mask] = np.exp(np.interp(logE, np.log(E_pa[1:]), np.log(mu_pa[1:])))  
    return res
def mu_coh_v(E):
    logE = np.log(np.clip(E, E_co[0], E_co[-1]))
    return np.exp(np.interp(logE, np.log(E_co), np.log(mu_co)))
def mu_total_v(E):
    return mu_comp_v(E)+mu_photo_v(E)+mu_pair_v(E)+mu_coh_v(E)

def sample_compton_v(E, rng):
    n = len(E)
    mu = np.empty(n)
    alive = np.ones(n, bool)
    a = E/me
    while alive.any():
        idx = np.where(alive)[0]
        mu_try = rng.uniform(-1, 1, len(idx))
        f = kn_f(mu_try, E[idx])
        acc = rng.uniform(0, 2, len(idx)) < f
        mu[idx[acc]] = mu_try[acc]
        alive[idx[acc]] = False
    Ep = E / (1.0 + a*(1.0-mu))
    te = E - Ep
    num = E - Ep*mu
    den = np.sqrt(E*E + Ep*Ep - 2.0*E*Ep*mu)
    cos_el = np.where(den > 1e-15, num/den, 1.0)
    return mu, Ep, te, cos_el

def rotate_dir_v(ux, uy, uz, mu_s, phi):
    d = np.column_stack([ux, uy, uz])
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / norm
    ref = np.tile(np.array([0.0,0.0,1.0]), (len(d),1))
    near_z = np.abs(d[:,2]) > 0.9
    ref[near_z] = [1.0, 0.0, 0.0]
    e1 = np.cross(d, ref)
    nrm1 = np.linalg.norm(e1, axis=1, keepdims=True)
    e1 = e1 / (nrm1 + 1e-30)
    e2 = np.cross(d, e1)
    st = np.sqrt(np.clip(1.0 - mu_s*mu_s, 0, 1))
    mu_s2 = mu_s[:, np.newaxis]
    st2 = st[:, np.newaxis]
    phi2 = phi[:, np.newaxis]
    newd = st2*np.cos(phi2)*e1 + st2*np.sin(phi2)*e2 + mu_s2*d
    newd = newd / np.linalg.norm(newd, axis=1, keepdims=True)
    return newd[:,0], newd[:,1], newd[:,2]

def inside_cube_v(x,y,z):
    return (np.abs(x)<15)&(np.abs(y)<15)&(>100)&(3<130)

def in_scoring_v(x,y,z):
    return (x*x+y*y<1.0)&(>109.9)&(3<110.1)

# recreate the simulate_batch function (same as before but using the above definitions)
def simulate_batch2(n_prim, rng, events_list):
    x = rng.uniform(-5,5,n_prim)
    y = rng.uniform(-5,5,n_prim)
    z = np.full(n_prim, 100.0)
    dist = np.sqrt(x*x + y*y + z*z)
    ux = x/dist; uy = y/dist; uz = z/dist
    E = np.full(n_prim, 2.0)
    alive = np.ones(n_prim, bool)
    extra_x = []; extra_y = []; extra_z = []
    extra_ux = []; extra_uy = []; extra_uz = []; extra_E = []
    
    for iteration in range(80):
        n = np.sum(alive)
        if n == 0:
            break
            idx = np.where(alive)[0]
        E_a = E[idx]; ux_a = ux[idx]; uy_a = uy[idx]; uz_a = uz[idx]
        x_a = x[idx]; y_a = y[idx]; z_a = z[idx]
        
        mu_c = mu_comp_v(E_a); mu_ph = mu_photo_v(E_a); mu_pa = mu_pair_v(E_a); mu_co = mu_coh_v(E_a)
        mu_tot = mu_c + mu_ph + mu_pa + mu_co
        
        s = -np.log(rng.uniform(1e-10,1,n)) / mu_tot
        x_a += s*ux_a        y_a += s*uy_a;        z_a += s*uz_a
        
        inside = inside_cube_v(x_a, y_a, z_a)
        in_idx = idx[inside]
        n_in = len(in_idx)
        
        if n_in > 0:
            E_i = E[in_idx]; mu_c_i = mu_c[inside]; mu_ph_i = mu_ph[inside]
            mu_pa_i = mu_pa[inside]; mu_co_i = mu_co[inside]; mu_tot_i = mu_tot[inside]
            x_i = x_a[inside]; y_i = y_a[inside]; z_i = z_a[inside]
            ux_i = ux_a[inside]; uy_i = uy_a[inside]; uz_i = uz_a[inside]
            scoring = in_scoring_v(x_i, y_i, z_i)
            
            u = rng.uniform(0, mu_tot_i, n_in)
            cum_c = mu_c_i; cum_ph = mu_c_i+mu_ph_i; cum_pa = mu_c_i+mu_ph_i+mu_pa_i
            proc = np.full(n_in, 3)
            proc[u <= cum_c] = 0
            proc[(u > cum_c)&(u <= cum_ph)] = 1
            proc[(u > cum_ph)&(u <= cum_pa)] = 2
            
            # compton
            mask_c = proc==0
            if mask_c.any():
                idx_c = in_idx[mask_c]; E_c = E_i[mask_c]
                mu_s, Ep, te, cos_el = sample_compton_v(E_c, rng)
                sc_c = scoring[mask_c]
                if sc_c.any():
                    for i in np.where(sc_c)[0]:
                        events_list.append({'process':'compton','e_in_MeV':float(E_c[i]),
                           'e_out_MeV':float(Ep[i]),'cos_scatter':float(mu_s[i]),
                           't_e_MeV':float(te[i]),'cos_electron':float(cos_el[i]),
                           't_positron_MeV':None})
                phi_c = rng.uniform(0,2*np.pi,len(E_c))
                ux_n, uy_n, uz_n = rotate_dir_v(ux_i[mask_c], uy_i[mask_c], uz_i[mask_c], mu_s, phi_c)
                E[idx_c] = Ep; ux[idx_c]=ux_n; uy[idx_c]=uy_n; uz[idx_c]=uz_n
                x[idx_c]=x_i[mask_c]; y[idx_c]=y_i[mask_c]; z[idx_c]=z_i[mask_c]
            
            # photo
            mask_pe = proc==1
            if mask_pe.any():
                idx_pe = in_idx[mask_pe]; E_pe = E_i[mask_pe]
                sc_pe = scoring[mask_pe]
                if sc_pe.any():
                    for i in np.where(sc_pe)[0]:
                        events_list.append({'process':'photoelectric','e_in_MeV':float(E_pe[i]),
                            'e_out_MeV':None,'cos_scatter':None,'t_e_MeV':float(E_pe[i]),
                            'cos_electron':float(rng.uniform(-1,1)),'t_positron_MeV':None})
                alive[ix_pe] = False
            
            # pair
            mask_pa = proc==2
            if mask_pa.any():
                idx_pa = in_idx[mask_pa]; E_pa = E_i[mask_pa]
                sc_pa = scoring[mask_pa]
                tavail = E_pa - 1.022; share = rng.uniform(0,1,len(E_pa))
                t_pos = share*tavail; t_e = tavail - t_pos
                E_tot = t_e + me; th0 = me/np.clip(E_tot,1e-10,None)
                th_e = th0 * (-np.log(rng.uniform(1e-10,1,len(E_pa))))
                cos_el = np.cos(th_e)
                if sc_pa.any():
                    for i in np.where(sc_pa)[0]:
                        events_list.append({'process':'pair','e_in_MeV':float(E_pa[i]),
                            'e_out_MeV':None,'cos_scatter':None,'t_e_MeV':float(t_e[i]),
                            'cos_electron':float(cos_el[i]),'t_positron_MeV':float(t_pos[i])})
                for i in range(len(E_pa)):
                    for _ in range(2):
                        u1 = rng.uniform(-1,1); u2 = rng.uniform(0,2*np.pi)
                        nz = u1; nx = math.sqrt(1-u1*u1)*math.cos(u2); ny = math.sqrt(1-u1*u1)*math.sin(u2)
                        extra_x.append(x_i[mask_pa][i]); extra_y.append(y_i[mask_pa][i]); extra_z.append(z_i[mask_pa][i])
                        extra_ux.append(nx); extra_uy.append(ny); extra_uz.append(nz); extra_E.append(0.511)
                alive[idx_pa] = False
            
            # rayleigh
            mask_r = proc==3
            if mask_r.any():
                idx_r = in_idx[mask_r]; E_r = E_i[mask_r]
                sc_r = scoring[mask_r]
                th0 = np.clip(0.0007/E_r,1e-6,0.5)
                th_r = th0*(-np.log(rng.uniform(1e-10,1,len(E_r))))
                mu_s_r = np.cos(th_r)
                if sc_r.any():
                    for i in np.where(sc_r)[0]:
                        events_list.append({'process':'rayleigh','e_in_MeV':float(E_r[i]),
                            'e_out_MeV':float(E_r[i]),'cos_scatter':float(mu_s_r[i]),
                            't_e_MeV':0.0,'cos_electron':None,'t_positron_MeV':None})
                phi_r = rng.uniform(0,2*np.pi,len(E_r))
                ux_n,uy_n,uz_n = rotate_dir_v(ux_i[mask_r],uy_i[mask_r],uz_i[mask_r],mu_s_r,phi_r)
                E[idx_r]=E_r; ux[idx_r]=ux_n; uy[idx_r]=uy_n; uz[idx_r]=uz_n
                x[idx_r]=x_i[mask_r]; y[idx_r]=y_i[mask_r]; z[idx_r]=z_i[mask_r]
        
        alive[in_idx] = True
        outside_idx = idx[~inside]
        alive[outside_idx] = False
        
        if len(extra_x)>0:
            x = np.concatenate([x,np.array(extra_x)])
            y = np.concatenate([y,np.array(extra_y)])
            z = np.concatenate([z,np.array(extra_z)])
            ux = np.concatenate([ux,np.array(extra_ux)])
            uy = np.concatenate([uy,np.array(extra_uy)])
            uz = np.concatenate([uz,np.array(extra_uz)])
            E = np.concatenate([E,np.array(extra_E)])
            alive = np.concatenate([alive,np.ones(len(extra_x),bool)])
            extra_x=[];extra_y=[];extra_z=[];extra_ux=[];extra_uy=[];extra_uz=[];extra_E=[]
        
        if np.sum(~alive)>len(alive)//2:
            keep=np.where(alive)[0]
            x=x[keep];y=y[keep];z=z[keep];ux=ux[keep];uy=uy[keep];uz=uz[keep];E=E[keep];alive=alive[keep]
    
    return

# Run
events = []
t0 = time.time()
simulate_batch2(200000, rng, events)
t1 = time.time()
print(f"200000 primaries: {len(events)} events in {t1-t0:.1f}s")
if events:
    cnt = Counter(e['process'] for e in events)
    print(cnt)