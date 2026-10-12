def mu_pair(E):
    res = np.zeros_like(E)
    mask = E>1.022
    if mask.any():
        logE = np.log(np.clip(E[mask],E_pa[1],E_pa[-1]))
        res[mask] = np.exp(np.interp(logE, np.log(E_pa[1:]), np.log(mu_pa[1:])))
    return res
def mu_coh(E):
    logE = np.log(np.clip(E,E_co[0],E_co[-1]))
    return np.exp(np.interp(logE, np.log(E_co), np.log(mu_co)))

def sample_compton(E, rng):
    n = len(E)
    mu = np.empty(n)
    alive = np.ones(n,bool)
    a = E/me
    while alive.any():
        idx = np.where(alive)[0]
        mu_try = rng.uniform(-1,1,len(idx))
        f = kn_f(mu_try, E[idx])
        acc = rng.uniform(0,2,len(idx)) < f
        mu[idx[acc]] = mu_try[acc]
        alive[idx[acc]] = False
    Ep = E / (1.0 + a*(1.0-mu))
    te = E - Ep
    num = E - Ep*mu
    den = np.sqrt(E*E + Ep*Ep - 2.0*E*Ep*mu)
    cos_el = np.where(den>1e-15, num/den, 1.0)
    return mu, Ep, te, cos_el

def rotate_dir(ux,uy,uz, mu_s, phi):
    d = np.column_stack([ux,uy,uz])
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    ref = np.tile(np.array([0.,0.,1.]),(len(d),1))
    nz = np.abs(d[:,2]) > 0.9
    ref[nz] = [1.,0.,0.]
    e1 = np.cross(d,ref)
    e1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-30)
    e2 = np.cross(d,e1)
    st = np.sqrt(np.clip(1.0-mu_s*mu_s,0,1))
    nd = (st*np.cos(phi))[:,np.newaxis]*e1 + (st*np.sin(phi))[:,np.newaxis]*e2 + mu_s[:,np.newaxis]*d
    nd = nd / np.linalg.norm(nd, axis=1, keepdims=True)
    return nd[:,0], nd[:,1], nd[:,2]

def inside_cube(x,y,z):
    return (np.abs(x)<15)&(np.abs(y)<15)&(z>100)&(z<130)
def in_scoring(x,y,z):
    return (x*x+y*y<1.0)&(z>109.9)&(z<110.1)

def run_batch(n_prim, events_out):
    x = rng.uniform(-5,5,n_prim)
    y = rng.uniform(-5,5,n_prim)
    z = np.full(n_prim, 100.0)
    dist = np.sqrt(x*x+y*y+z*z)
    ux = x/dist; uy = y/dist; uz = z/dist
    E = np.full(n_prim, 2.0)
    alive = np.ones(n_prim,bool)
    extra = {'x':[],'y':[],'z':[],'ux':[],'uy':[],'uz':[],'E':[]}
    
    for _ in range(80):
        n = np.sum(alive)
        if n==0: break
        idx = np.where(alive)[0]
        E_a = E[idx]; ux_a=ux[idx]; uy_a=uy[idx]; uz_a=uz[idx]
        x_a=x[idx]; y_a=y[idx]; z_a=z[idx]
        
        mu_c = mu_comp(E_a); mu_ph = mu_photo(E_a); mu_pa = mu_pair(E_a); mu_co = mu_coh(E_a)
        mu_all = mu_c+mu_ph+mu_pa+mu_co
        
        s = -np.log(rng.uniform(1e-10,1,n)) / mu_all
        x_a += s*ux_a; y_a += s*uy_a; z_a += s*uz_a
        
        inside = inside_cube(x_a,y_a,z_a)
        in_idx = idx[inside]
        n_in = len(in_idx)
        
        if n_in>0:
            E_i = E[in_idx]; ux_i=ux_a[inside]; uy_i=uy_a[inside]; uz_i=uz_a[inside]
            x_i=x_a[inside]; y_i=y_a[inside]; z_i=z_a[inside]
            mu_c_i=mu_c[inside]; mu_ph_i=mu_ph[inside]; mu_pa_i=mu_pa[inside]; mu_co_i=mu_co[inside]; mu_all_i=mu_all[inside]
            
            scoring = in_scoring(x_i,y_i,z_i)
            
            u = rng.uniform(0, mu_all_i, n_in)
            cum_c = mu_c_i; cum_ph = mu_c_i+mu_ph_i; cum_pa = mu_c_i+mu_ph_i+mu_pa_i
            proc = np.full(n_in, 3)
            proc[u<=cum_c]=0; proc[(u>cum_c)&(u<=cum_ph)]=1; proc[(u>cum_ph)&(u<=cum_pa)]=2
            
            # ... need to be very careful. This is too error-prone.
            pass

print("Functions defined")