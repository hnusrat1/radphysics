
# Full photon MC, multiple depths; save results
import numpy as np, time, pickle

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
def mu_pair(E): return np.where(E>1.022, 0.00052*((E-1.022)/0.978)**2.0, 0.0)
def mu_pe(E): return np.maximum(0, mu_tot(E)-mu_C(E)-mu_coh(E)-mu_pair(E))
def mu_tot(E): return interp(E,E_tab,mu_tab)

def kn_diff(cosT, E):
    al = E/mec2; x = 1+al*(1-cosT)
    return (1.0/x)**2*(0.5*(1+cosT*cosT)+(al*al*(1-cosT)**2)/(2*x))
def sample_compton(E, rng):
    x = rng.uniform(-1,1,size=E.shape)
    acc = rng.uniform(0,1,size=E.shape) < kn_diff(x,E)
    while not acc.all():
        rem = ~acc
        nx = rng.uniform(-1,1,size=rem.sum())
        nacc = rng.uniform(0,1,size=rem.sum()) < kn_diff(nx, E[rem])
        x[rem] = nx; acc[rem] = nacc
    al = E/mec2
    Ep = E/(1+al*(1-x))
    Ee = E - Ep
    pe = np.sqrt(np.maximum(Ee*(Ee+2*mec2),0))
    cosphi = np.clip((E - Ep*x)/pe, -1, 1)
    return Ep, x, cosphi
def rotate_dir(u, cosT, rng):
    sinT = np.sqrt(np.maximum(1-cosT*cosT,0))
    temp = rng.normal(size=u.shape)
    ndot = np.sum(temp*u,axis=1,keepdims=True)
    n = temp - ndot*u
    nnorm = np.linalg.norm(n,axis=1,keepdims=True)
    n = n/(nnorm+1e-12)
    m = np.cross(u,n)
    cosphi = rng.uniform(-1,1,size=len(u))
    sinphi = np.sqrt(np.maximum(1-cosphi**2,0))
    new = cosT[:,None]*u + sinT[:,None]*(cosphi[:,None]*m + sinphi[:,None]*n)
    return new/np.linalg.norm(new,axis=1,keepdims=True)
def ray_cylinder_contrib(p, u, E, z0=5.0, R=1.0, h=0.1, V=0.6283185):
    N = len(p)
    if N==0: return np.zeros(0)
    a = u[:,0]**2 + u[:,1]**2
    b = 2*(p[:,0]*u[:,0] + p[:,1]*u[:,1])
    c = p[:,0]**2 + p[:,1]**2 - R**2
    disc = b*b - 4*a*c
    t_in = np.full(N, np.inf); t_out = np.full(N, -np.inf)
    m = a>1e-12
    sd = np.sqrt(np.maximum(disc,0))
    t1 = np.where(m, (-b-sd)/(2*a), np.inf)
    t2 = np.where(m, (-b+sd)/(2*a), np.inf)
    for t in [t1,t2]:
        z = p[:,2]+t*u[:,2]
        ok = (z>=z0-h)&(z<=z0+h)&(t>=-1e-7)
        tv = np.maximum(t,0)
        t_in = np.minimum(t_in, np.where(ok, tv, np.inf))
        t_out = np.maximum(t_out, np.where(ok, tv, -np.inf))
    for zc in [z0-h, z0+h]:
        tz = (zc - p[:,2])/u[:,2]
        xz = p[:,0]+tz*u[:,0]; yz=p[:,1]+tz*u[:,1]
        ok = (xz**2+yz**2 <= R*R+1e-10)&(tz>=-1e-7)
        t_in = np.minimum(t_in, np.where(ok, np.maximum(tz,0), np.inf))
        t_out = np.maximum(t_out, np.where(ok, np.maximum(tz,0), -np.inf))
    hit = t_in < np.inf
    r2 = p[:,0]**2+p[:,1]**2
    inside = (r2<=R*R)&(p[:,2]>=z0-h)&(p[:,2]<=z0+h)
    t_in = np.where(inside, 0.0, t_in)
    s_surf = np.where(p[:,2]<0, (0-p[:,2])/np.maximum(u[:,2],1e-12), 0.0)
    s_w = np.maximum(t_in - s_surf, 0.0)
    L = np.maximum(t_out-t_in,0)
    mu = mu_tot(E)
    cont = np.where(hit&(L>1e-9), np.exp(-mu*s_w)*(1-np.exp(-mu*L))/mu, 0.0)
    return cont/V

def run_MC(N, z0s, seed=12345, chunk=300000):
    rng = np.random.default_rng(seed)
    nz = len(z0s)
    ph_bins = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
    ct_bins = np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
    ebin = lambda E: np.clip(np.searchsorted(ph_bins, E, side='right')-1, 0, len(ph_bins)-2)
    ctbin = lambda ct: np.clip(np.searchsorted(ct_bins, ct, side='right')-1, 0, len(ct_bins)-2)
    prim_flu = np.zeros(nz); scat_flu = np.zeros(nz)
    prim_spec = np.zeros((nz,len(ph_bins)-1)); scat_spec = np.zeros((nz,len(ph_bins)-1))
    scat_ang = np.zeros(len(ct_bins)-1); prim_ang = np.zeros(len(ct_bins)-1)
    n_hist = 0; n_sc_hit=0; n_sc_seg=0
    t0=time.time()
    while n_hist < N:
        n = min(chunk, N-n_hist)
        E = rng.choice([1.1732,1.3325], size=n)
        dx = rng.uniform(-5,5,size=n); dy = rng.uniform(-5,5,size=n)
        vec = np.stack([dx,dy,np.full(n,100.0)],axis=1)
        u = vec/np.linalg.norm(vec,axis=1,keepdims=True)
        pos_src = np.zeros((n,3)); pos_src[:,2]=-100.0
        for iz,z0 in enumerate(z0s):
            cont = ray_cylinder_contrib(pos_src,u,E,z0)
            prim_flu[iz]+=cont.sum()
            np.add.at(prim_spec[iz], ebin(E), cont)
            if z0==5.0:
                np.add.at(prim_ang, ctbin(u[:,2]), cont)
        pos = np.zeros((n,3)); pos[:,0]=dx; pos[:,1]=dy
        primary = np.ones(n,bool); alive = np.ones(n,bool)
        while alive.any():
            sc = (~primary)&alive
            if sc.any():
                n_sc_seg += sc.sum()
                for iz,z0 in enumerate(z0s):
                    cont = ray_cylinder_contrib(pos[sc],u[sc],E[sc],z0)
                    scat_flu[iz]+=cont.sum()
                    np.add.at(scat_spec[iz], ebin(E[sc]), cont)
                    if z0==5.0:
                        np.add.at(scat_ang, ctbin(u[sc,2]), cont)
                        n_sc_hit += int((cont>0).sum())
            idx_alive = np.nonzero(alive)[0]
            muE = mu_tot(E[idx_alive])
            s = -np.log(rng.uniform(1e-300,1,size=len(idx_alive)))/muE
            p = pos[idx_alive]; uu = u[idx_alive]
            tz = np.where(uu[:,2]>0,(30-p[:,2])/uu[:,2], np.where(uu[:,2]<0,(0-p[:,2])/uu[:,2], np.inf))
            tx = np.where(uu[:,0]>0,(15-p[:,0])/uu[:,0], np.where(uu[:,0]<0,(-15-p[:,0])/uu[:,0], np.inf))
            ty = np.where(uu[:,1]>0,(15-p[:,1])/uu[:,1], np.where(uu[:,1]<0,(-15-p[:,1])/uu[:,1], np.inf))
            texit = np.minimum(np.minimum(tx,ty),tz)
            exitmask = s>=texit
            step = np.minimum(s,texit)
            pnew = p + step[:,None]*uu
            proc_alive = ~exitmask
            if proc_alive.any():
                Ecur = E[idx_alive][proc_alive]
                muC = mu_C(Ecur); muP = mu_pe(Ecur); muPA = mu_pair(Ecur); muR = mu_coh(Ecur)
                mtot = muC+muP+muPA+muR
                rp = rng.uniform(0,1,size=proc_alive.sum())
                wC = muC/mtot
                isC = rp<wC
                isPE = (~isC)&(rp<wC+muP/mtot)
                isPA = (~isC)&(~isPE)&(rp<wC+muP/mtot+muPA/mtot)
                isR = ~(isC|isPE|isPA)
                idx_proc = idx_alive[proc_alive]
                if isC.any():
                    Ep, cT, cpe = sample_compton(Ecur[isC], rng)
                    E[idx_proc[isC]] = Ep
                    u[idx_proc[isC]] = rotate_dir(u[idx_proc[isC]], cT, rng)
                    primary[idx_proc[isC]] = False
                if isR.any():
                    sz = isR.sum()
                    ctr = rng.uniform(-1,1,size=sz)
                    acc = rng.uniform(0,1,size=sz) < (1+ctr**2)/2
                    while not acc.all():
                        rem = ~acc
                        nw = rng.uniform(-1,1,size=rem.sum())
                        nacc = rng.uniform(0,1,size=rem.sum()) < (1+nw**2)/2
                        ctr[rem]=nw; acc[rem]=nacc
                    u[idx_proc[isR]] = rotate_dir(u[idx_proc[isR]], ctr, rng)
                    primary[idx_proc[isR]] = False
                if isPA.any():
                    npa = isPA.sum()
                    th = rng.uniform(0,np.pi,size=npa); ph = rng.uniform(0,2*np.pi,size=npa)
                    d1 = np.stack([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)],axis=1)
                    d2 = -d1
                    ppa = pos[idx_proc[isPA]]
                    E = np.concatenate([E, np.full(2*npa,0.511)]); primary = np.concatenate([primary, np.zeros(2*npa,bool)])
                    u = np.concatenate([u, d1, d2],axis=0); pos = np.concatenate([pos, ppa, ppa],axis=0)
                    alive = np.concatenate([alive, np.ones(2*npa,bool)])
                pos[idx_proc] = pnew[proc_alive]
                alive[idx_proc[isPE|isPA]] = False
            alive[idx_alive[exitmask]] = False
        n_hist += n
        print(f"  {n_hist}/{N} elapsed {time.time()-t0:.1f}s", flush=True)
    return dict(N=N, prim_flu=prim_flu, scat_flu=scat_flu, prim_spec=prim_spec, scat_spec=scat_spec,
                prim_ang=prim_ang, scat_ang=scat_ang, n_sc_hit=n_sc_hit, n_sc_seg=n_sc_seg)

z0s = np.array([0.3,0.5,0.7,1.0,1.5,2.0,3.0,4.0,5.0,6.0,8.0,10.0,12.0,15.0,20.0])
N = 3000000
res = run_MC(N=N, z0s=z0s, seed=20240517)
with open('photon_mc.pkl','wb') as f:
    pickle.dump({'z0s':z0s,'N':N,'res':res}, f)
phi0ph = 1/(0.01*105**2)
print("n_sc_seg:", res['n_sc_seg'], "n_sc_hit:", res['n_sc_hit'])
print("scatter hit rate:", res['n_sc_hit']/res['n_sc_seg'])
print("z0, prim/phi0, scat/phi0:")
for i,z0 in enumerate(z0s):
    print(f"  {z0:5.1f}  {res['prim_flu'][i]/(N*phi0ph):.4f}  {res['scat_flu'][i]/(N*phi0ph):.4f}")
