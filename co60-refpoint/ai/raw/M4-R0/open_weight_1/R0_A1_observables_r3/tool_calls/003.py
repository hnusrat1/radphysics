import numpy as np, time

mec2=0.5110
N_e_per_g = 6.02214076e23*10/18.01528
E_tab = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0])
mu_tab = np.array([4.951,1.374,0.7031,0.4470,0.2269,0.2059,0.1837,0.1706,0.1505,0.1370,0.1186,0.1061,0.0969,0.0896,0.0786,0.0707,0.0636,0.0576,0.0494])
coh_tab = np.array([0.0,0.0,0.0,0.03,0.0205,0.0148,0.0102,0.0084,0.0047,0.0030,0.0014,0.0009,0.0006,0.00045,0.00030,0.00024,0.00019,0.00016,0.00010])
def interp(x,xe,ye): return np.interp(x,xe,ye)
def sig_KN(E):
    al=E/mec2; a2=al*al
    f=(1+al)/a2*(2*(1+al)/(1+2*al)-np.log(1+2*al)/al)+np.log(1+2*al)/(2*al)-(1+3*al)/(1+2*al)**2
    return 0.4987*f
def mu_C(E): return sig_KN(E)*1e-24*N_e_per_g
def mu_coh(E): return interp(E,E_tab,coh_tab)
def mu_pair(E):
    return np.where(E>1.022, 0.00052*((E-1.022)/0.978)**2.0, 0.0)
def mu_pe(E):
    return np.maximum(0, mu_tot(E)-mu_C(E)-mu_coh(E)-mu_pair(E))
def mu_tot(E): return interp(E,E_tab,mu_tab)
def mu_tot_parts(E): return mu_C(E)+mu_pe(E)+mu_pair(E)+mu_coh(E)

for E in [0.1,0.5,1.1732,1.25,1.3325,1.4,2.0]:
    print(f"E={E:.4f} tot={mu_tot(E):.4f} C={mu_C(E):.4f} pe={mu_pe(E):.5f} pair={mu_pair(E):.5f} coh={mu_coh(E):.5f}")

# ---------- geometry ----------
# source at (0,0,-100), surface z=0, phantom z in [0,30], x,y in [-15,15]
# cylinder z0, R=1, half-thickness 0.1

def ray_cylinder_contrib(p, u, E, z0, R=1.0, h=0.1, V=0.6283185):
    N = len(p)
    # returns contributions (N,) using full track-length estimator
    a = u[:,0]**2 + u[:,1]**2
    b = 2*(p[:,0]*u[:,0] + p[:,1]*u[:,1])
    c = p[:,0]**2 + p[:,1]**2 - R**2
    disc = b*b - 4*a*c
    t_in = np.full(N, np.inf); t_out = np.full(N, -np.inf)
    # side candidates
    m = a>1e-12
    sd = np.sqrt(np.maximum(disc,0))
    t1 = (-b - sd)/(2*abs(a)+1e-30); t2 = (-b + sd)/(2*abs(a)+1e-30)
    t1 = np.where(m, (-b-sd)/(2*a), np.inf); t2 = np.where(m, (-b+sd)/(2*a), np.inf)
    for t in [t1,t2]:
        z = p[:,2]+t*u[:,2]
        ok = (z>=z0-h)&(z<=z0+h)&(t>=-1e-7)
        t_in = np.minimum(t_in, np.where(ok, np.maximum(t,0), np.inf))
        t_out = np.maximum(t_out, np.where(ok, np.maximum(t,0), -np.inf))
    # caps
    for zc in [z0-h, z0+h]:
        tz = (zc - p[:,2])/u[:,2]
        xz = p[:,0]+tz*u[:,0]; yz=p[:,1]+tz*u[:,1]
        ok = (xz**2+yz**2 <= R*R+1e-10)&(tz>=-1e-7)
        t_in = np.minimum(t_in, np.where(ok, np.maximum(tz,0), np.inf))
        t_out = np.maximum(t_out, np.where(ok, np.maximum(tz,0), -np.inf))
    hit = t_in < np.inf
    # if photon inside cylinder: t_in=0
    r2 = p[:,0]**2+p[:,1]**2
    inside = (r2<=R*R)&(p[:,2]>=z0-h)&(p[:,2]<=z0+h)
    t_in = np.where(inside, 0.0, t_in)
    # water path: subtract distance to surface (pz<0)
    s_surf = np.where(p[:,2]<0, (0-p[:,2])/np.maximum(u[:,2],1e-12), 0.0)
    s_w = np.where(hit, np.maximum(t_in - s_surf, 0.0), 1e30)
    L = np.where(hit, np.maximum(t_out-t_in,0), 0.0)
    mu = mu_tot(E)
    # small-L approx: (1-exp(-mu L))/mu = L for small
    cont = np.where(hit&(L>1e-9), np.exp(-mu*s_w)*(1-np.exp(-mu*L))/mu, 0.0)
    return cont/V, hit

# ---------- KN sampling ----------
def sample_compton(E, rng):
    # returns E', cosT(photon), cosphi(electron)
    al = E/mec2
    x = rng.uniform(-1,1,size=E.shape)
    # rejection
    accept = rng.uniform(0,1,size=E.shape) < kn_diff(x,E)/1.0
    while not accept.all():
        newx = rng.uniform(-1,1,size=accept.size)
        newacc = rng.uniform(0,1,size=accept.size) < kn_diff(newx,E[accept])/1.0
        x[accept] = newx[newacc]
        accept[accept] = newacc
    Ep = E/(1+al*(1-x))
    Ee = E - Ep
    pe = np.sqrt(Ee*(Ee+2*mec2))
    cosphi = np.clip((E - Ep*x)/pe, -1, 1)
    return Ep, x, cosphi

def kn_diff(cosT, E):
    al = E/mec2
    x = 1+al*(1-cosT)
    return (1.0/x)**2*(0.5*(1+cosT*cosT)+(al*al*(1-cosT)**2)/(2*x))

def rotate_dir(u, cosT, rng):
    # rotate unit vector u by polar angle theta around random azimuth
    sinT = np.sqrt(np.maximum(1-cosT*cosT,0))
    # axis: random vector perpendicular to u
    temp = rng.normal(size=u.shape)
    # n = normalize(temp - (temp.u)u)
    ndot = np.sum(temp*u,axis=1,keepdims=True)
    n = temp - ndot*u
    nnorm = np.linalg.norm(n,axis=1,keepdims=True)
    n = n/ (nnorm+1e-12)
    m = np.cross(u,n)
    cosphi = rng.uniform(-1,1,size=len(u))
    sinphi = np.sqrt(np.maximum(1-cosphi**2,0))
    new = cosT[:,None]*u + sinT[:,None]*(cosphi[:,None]*m + sinphi[:,None]*n)
    new = new/np.linalg.norm(new,axis=1,keepdims=True)
    return new

# test
rng = np.random.default_rng(1)
Etest = np.full(20000,1.25)
Ep, xt, cp = sample_compton(Etest,rng)
print("mean Ep",Ep.mean(),"expected ~",1.25/(1+ (1.25/0.511)*0.5), "mean Ee", (Etest-Ep).mean())
print("mean cosphi", cp.mean())

# ---------- photon transport ----------
def run_MC(N=200000, z0s=np.array([5.0]), seed=42):
    rng = np.random.default_rng(seed)
    E_lines = np.array([1.1732,1.3325])
    nz = len(z0s)
    # tallies: primary fluence per z, scattered fluence per z, energy bins, angular at z=5
    ph_bins = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
    ct_bins = np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
    def ebin(E): return np.searchsorted(ph_bins, E, side='right')-1
    def ctbin(ct): return np.searchsorted(ct_bins, ct, side='right')-1
    prim_flu = np.zeros(nz)
    scat_flu = np.zeros(nz)
    prim_spec = np.zeros((nz,len(ph_bins)-1))
    scat_spec = np.zeros((nz,len(ph_bins)-1))
    scat_ang_z5 = np.zeros(len(ct_bins)-1)
    prim_ang_z5 = np.zeros(len(ct_bins)-1)
    
    chunk = min(N, 500000)
    done = 0
    t0=time.time()
    while done < N:
        n = min(chunk, N-done)
        # init
        E = rng.choice(E_lines, size=n)
        dx = rng.uniform(-5,5,size=n); dy = rng.uniform(-5,5,size=n)
        vec = np.stack([dx,dy,np.full(n,100.0)],axis=1)
        u = vec/np.linalg.norm(vec,axis=1,keepdims=True)
        pos = np.zeros((n,3)); pos[:,2] = -100.0
        primary = np.ones(n, bool)   # True if this photon is a primary (unscattered)
        alive = np.ones(n, bool)
        # estimator at source segment
        for iz,z0 in enumerate(z0s):
            cont,_ = ray_cylinder_contrib(pos, u, E, z0)
            prim_flu[iz] += cont.sum()
            bi = ebin(E); bi[bi<0]=0; bi[bi>=len(ph_bins)-1]=len(ph_bins)-2
            np.add.at(prim_spec[iz], bi, cont)
            if z0==5.0:
                cbi = ctbin(u[:,2]); cbi[cbi<0]=0; cbi[cbi>=10]=9
                np.add.at(prim_ang_z5, cbi, cont)
        # transport loop
        while alive.any():
            # add estimator for scattered (post-interaction) segments: primary flag False or those that have interacted
            # (do it for all alive; primary contribution already counted once at source, so only scatter flag)
            sc = (~primary) & alive
            if sc.any():
                for iz,z0 in enumerate(z0s):
                    cont,_ = ray_cylinder_contrib(pos[sc], u[sc], E[sc], z0)
                    scat_flu[iz] += cont.sum()
                    bi = ebin(E[sc]); bi[bi<0]=0; bi[bi>=len(ph_bins)-1]=len(ph_bins)-2
                    np.add.at(scat_spec[iz], bi, cont)
                    if z0==5.0:
                        cbi = ctbin(u[sc,2]); cbi[cbi<0]=0; cbi[cbi>=10]=9
                        np.add.at(scat_ang_z5, cbi, cont)
            # step
            muE = mu_tot(E[alive])
            s = -np.log(rng.uniform(1e-300,1,size=alive.sum()))/muE
            # exit distance from current position
            p = pos[alive]; uu = u[alive]
            tz = np.where(uu[:,2]>0, (30-p[:,2])/uu[:,2], np.where(uu[:,2]<0, (0-p[:,2])/uu[:,2], np.inf))
            tx = np.where(uu[:,0]>0, (15-p[:,0])/uu[:,0], np.where(uu[:,0]<0, (-15-p[:,0])/uu[:,0], np.inf))
            ty = np.where(uu[:,1]>0, (15-p[:,1])/uu[:,1], np.where(uu[:,1]<0, (-15-p[:,1])/uu[:,1], np.inf))
            texit = np.minimum(np.minimum(tx,ty),tz)
            # if s crosses surface -> photon exits before interacting
            exitmask = s >= texit
            # move to interaction point (clamp) 
            step = np.minimum(s, texit)
            pnew = p + step[:,None]*uu
            # particles that exit: dead
            dead_out = exitmask
            # for survivors, process interaction
            proc_alive = ~dead_out
            # sample process for survivors
            if proc_alive.any():
                Ecur = E[alive][proc_alive]
                muC = mu_C(Ecur); muP = mu_pe(Ecur); muPA = mu_pair(Ecur); muR = mu_coh(Ecur)
                mtot = muC+muP+muPA+muR
                rp = rng.uniform(0,1,size=proc_alive.sum())
                # order: compton, pe, pair, rayleigh
                wC = muC/mtot
                isC = rp < wC
                isPE = (~isC) & (rp < wC+muP/mtot)
                isPA = (~isC)&(~isPE)&(rp < wC+muP/mtot+muPA/mtot)
                isR = ~(isC|isPE|isPA)
                # apply
                idx_alive = np.nonzero(alive)[0]
                idx_proc = idx_alive[proc_alive]
                # compton
                if isC.any():
                    Ep, ct, cpe = sample_compton(Ecur[isC], rng)
                    # replace
                    E[idx_proc[isC]] = Ep
                    u[idx_proc[isC]] = rotate_dir(u[idx_proc[isC]], ct, rng)
                    primary[idx_proc[isC]] = False
                if isR.any():
                    # Rayleigh: forward small angle: sample cosT from Thomson pdf sinT*(1+cos^2) via rejection
                    sz = isR.sum()
                    ctr = rng.uniform(-1,1,size=sz)
                    # pdf (1+cos^2)/2 normalized 1; simple accept: (1+c^2)/2
                    acc = rng.uniform(0,1,size=sz) < (1+ctr**2)/2
                    while not acc.all():
                        nw = rng.uniform(-1,1,size=(~acc).sum())
                        nacc = rng.uniform(0,1,size=(~acc).sum()) < (1+nw**2)/2
                        ctr[acc] = nw[nacc]
                        acc[acc] = nacc
                    u[idx_proc[isR]] = rotate_dir(u[idx_proc[isR]], ctr, rng)
                    primary[idx_proc[isR]] = False
                # PE and PA: dead
                # for PA, add two annihilation photons
                if isPA.any():
                    npa = isPA.sum()
                    Enew = np.full(npa,0.511)
                    # random directions opposite
                    th = rng.uniform(0,np.pi,size=npa); ph = rng.uniform(0,2*np.pi,size=npa)
                    d1 = np.stack([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)],axis=1)
                    d2 = -d1
                    pnew_pa = pos[idx_proc[isPA]]
                    # append
                    E = np.concatenate([E, Enew, Enew])
                    u = np.concatenate([u, d1, d2],axis=0)
                    pos = np.concatenate([pos, pnew_pa, pnew_pa],axis=0)
                    primary = np.concatenate([primary, np.zeros(2*npa,bool)])
                    alive = np.concatenate([alive, np.ones(2*npa,bool)])
                # kill PE and PA (and mark moved positions for all processed)
            pos[idx_alive[proc_alive]] = pnew[proc_alive]
            # dead flags
            alive[idx_alive[exitmask]] = False
            alive[idx_alive[proc_alive & (isPE | isPA)]] = False
        done += n
        if done % 200000 == 0:
            print(f"  {done}/{N} histories, elapsed {time.time()-t0:.1f}s", flush=True)
    return dict(prim_flu=prim_flu, scat_flu=scat_flu, prim_spec=prim_spec, scat_spec=scat_spec,
                prim_ang=prim_ang_z5, scat_ang=scat_ang_z5, E_lines=E_lines)

if __name__ == '__main__':
    N=300000
    z0s = np.array([5.0])
    res = run_MC(N=N, z0s=z0s, seed=7)
    phi0ph = 1/(0.01*105**2)
    print("phi0 per history:", phi0ph)
    print("prim flu/phi0:", res['prim_flu'][0]/(N*phi0ph), "analytic:", 0.5*(np.exp(-5*mu_tot(1.1732))+np.exp(-5*mu_tot(1.3325))))
    print("scat flu/phi0:", res['scat_flu'][0]/(N*phi0ph))
    tot = (res['prim_flu'][0]+res['scat_flu'][0])/(N*phi0ph)
    print("total flu/phi0:", tot, " scatter fraction:", res['scat_flu'][0]/(res['prim_flu'][0]+res['scat_flu'][0]))
    print("scat angular norm:", res['scat_ang'].sum()/ (N*phi0ph), "scat flu:", res['scat_flu'][0]/(N*phi0ph))
