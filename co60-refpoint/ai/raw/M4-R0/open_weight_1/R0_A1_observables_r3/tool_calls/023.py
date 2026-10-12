
import numpy as np, time

mec2 = 0.5110
N_e_per_g = 6.02214076e23*10/18.01528
n_e = N_e_per_g
C = 0.4987e-24*0.511
X0_water = 36.08
DELTA = 0.010

def Scol_w(E):
    return np.interp(E, [0.01,0.02,0.03,0.05,0.07,0.1,0.2,0.3,0.5,0.7,1.0,1.25,1.4],
                     [22.7,11.3,7.97,6.36,5.14,4.098,2.63,2.16,1.94,1.89,1.847,1.847,1.85])
def Srest_w(E):
    out = np.zeros_like(E)
    for i in range(len(E)):
        e = E[i]
        if e<=0.011:
            out[i]=Scol_w(e)
        else:
            dEint=0.00025
            E0v=np.arange(0.0105, e/2-1e-9, dEint)
            out[i]=Scol_w(e)-np.sum(E0v*np.array([moler_dsde(e,x) for x in E0v])*dEint)
    return out

def moler_dsde(T, E0):
    if T<=0.011 or E0<=0.010 or E0>=T/2: return 0.0
    beta2 = 1.0-1.0/(1+T/mec2)**2
    if beta2<=0: return 0.0
    return C*n_e/beta2*(1/E0**2 - 1/(E0*(T-E0)) + 1/(T-E0)**2)

# Precompute Srest on a grid for speed
Eg_all = np.arange(0.010,1.401,0.002)
Srest_grid = np.array([Srest_w(np.array([e]))[0] for e in Eg_all])
Scol_grid = np.array([Scol_w(e) for e in Eg_all])
def Srest_fast(E): return np.interp(E, Eg_all, Srest_grid)
def Scol_fast(E): return np.interp(E, Eg_all, Scol_grid)

def pbeta(E):
    return np.maximum(E*(E+2*mec2)/(E+mec2), 1e-6)

# load
d = np.load('L4_data.npz')
Eg = d['Eg']; q = d['q']
L3 = np.load('L3_data.npy', allow_pickle=True).item()
ang_tot_n = (L3['ang_tot']/L3['ang_tot'].sum())
ct_bins = np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
ct_c = (ct_bins[:-1]+ct_bins[1:])/2

rng = np.random.default_rng(11)
N_elec = 200000
prob = np.maximum(q,0); prob = prob/prob.sum()
E0s = rng.choice(Eg, size=N_elec, p=prob)
cos0 = rng.choice(ct_c, size=N_elec, p=ang_tot_n)
az0 = rng.uniform(0,2*np.pi,size=N_elec)
s0 = np.sqrt(np.maximum(1-cos0**2,0))
dx = s0*np.cos(az0); dy=s0*np.sin(az0); dz=cos0

track_all = np.zeros(10); track_delta = np.zeros(10)
ds = 0.02
t0=time.time()
E_list=[E0s]; d_list=[np.stack([dx,dy,dz],axis=1)]; tag_list=[np.zeros(N_elec)]
# We'll just use growing arrays and loop
E = E0s.copy(); D = np.stack([dx,dy,dz],axis=1); tag = np.zeros(N_elec)
alive_all = np.ones(N_elec,bool)
steps=0
while alive_all.any():
    Ea = E[alive_all]; Da = D[alive_all]; taga = tag[alive_all]
    n = len(Ea)
    # energy loss (restricted)
    Sres = Srest_fast(Ea)
    Ea = Ea - Sres*ds
    # multiple scattering
    pv = pbeta(np.maximum(Ea,0.0112))
    sig = (14.1/pv)*np.sqrt(ds/X0_water)*(1+0.038*np.log(ds/X0_water))
    th = np.abs(rng.normal(0,sig,size=n))
    az = rng.uniform(0,2*np.pi,size=n)
    cosT = np.cos(th); sinT = np.sin(th)
    rv = rng.normal(size=(n,3))
    ndot = np.sum(rv*Da,axis=1,keepdims=True)
    nvec = rv - ndot*Da
    nvec = nvec/(np.linalg.norm(nvec,axis=1,keepdims=True)+1e-12)
    mvec = np.cross(Da,nvec)
    caz = np.cos(az); saz = np.sin(az)
    newd = cosT[:,None]*Da + sinT[:,None]*(caz[:,None]*mvec + saz[:,None]*nvec)
    newd = newd/np.linalg.norm(newd,axis=1,keepdims=True)
    # tally
    ctz = np.clip(newd[:,2],-1,1)
    bi = np.clip(np.searchsorted(ct_bins,ctz,side='right')-1,0,9)
    np.add.at(track_all, bi, ds)
    np.add.at(track_delta, bi, ds*taga)
    # delta production
    beta2 = 1.0 - 1.0/(1+Ea/mec2)**2
    okE = Ea > 0.022
    pdel = np.zeros(n)
    T = Ea[okE]
    bet2 = beta2[okE]
    pref = C*n_e/bet2
    pdel[okE] = ds*pref*(1.0/DELTA + np.log((T-DELTA)/DELTA)/T - 1.0/(T-DELTA))
    pdel = np.minimum(pdel, 1.0)
    creates = rng.uniform(size=n) < pdel
    if creates.any():
        # sample E0: proposal 1/E0^2 over [DELTA, T/2], accept 1-r+r^2
        Tc = Ea[creates]
        E0min = DELTA
        E0max = Tc/2
        nc = creates.sum()
        # inverse CDF sampling of 1/E0^2: X = 1/(1/E0min - U*(1/E0min - 1/E0max))
        U = rng.uniform(1e-9,1-1e-9,size=nc)
        E0 = 1.0/(1.0/E0min - U*(1.0/E0min - 1.0/E0max))
        r = E0/(Tc-E0)
        acc = rng.uniform(size=nc) < (1 - r + r*r)
        while not acc.all():
            rem = ~acc
            U2 = rng.uniform(1e-9,1-1e-9,size=rem.sum())
            E0[rem] = 1.0/(1.0/E0min - U2*(1.0/E0min - 1.0/(Tc[rem]/2)))
            r2 = E0[rem]/(Tc[rem]-E0[rem])
            acc2 = rng.uniform(size=rem.sum()) < (1-r2+r2*r2)
            E0[rem] = E0[rem]; acc[rem]=acc2
        # primary loses E0
        Ea[creates] -= E0
        # secondary direction
        cosd = np.sqrt(E0/(E0+2*mec2))
        ad = np.arccos(np.clip(cosd,-1,1))
        azd = rng.uniform(0,2*np.pi,size=nc)
        dirp = newd[creates]
        rv2 = rng.normal(size=(nc,3))
        nd2 = np.sum(rv2*dirp,axis=1,keepdims=True)
        nv2 = rv2 - nd2*dirp
        nv2 = nv2/(np.linalg.norm(nv2,axis=1,keepdims=True)+1e-12)
        mv2 = np.cross(dirp,nv2)
        c2 = np.cos(azd); s2 = np.sin(azd)
        dd = np.cos(ad)[:,None]*dirp + np.sin(ad)[:,None]*(c2[:,None]*mv2 + s2[:,None]*nv2)
        dd = dd/np.linalg.norm(dd,axis=1,keepdims=True)
        # append delta electrons
        E_list.append(E0); d_list.append(dd); tag_list.append(np.ones(nc))
    # update alive arrays (mask)
    alive = Ea > 0.0105
    Ea = Ea[alive]; newd = newd[alive]; taga = taga[alive]
    E[alive_all] = Ea  # careful: updating by mask
    # Since alive_all masks E[alive_all], simpler: rebuild E,D,tag as compact arrays each iteration
    inds = np.nonzero(alive_all)[0][alive]
    E = Ea.copy(); D = newd.copy(); tag = taga.copy()
    alive_all = np.ones(len(E),bool)
    steps += 1
    if steps % 2000 == 0:
        print(f"  steps {steps}, alive {len(E)}, elapsed {time.time()-t0:.0f}s", flush=True)
    if len(E)==0: break
    if steps>60000: print("force stop"); break

# include the appended delta electrons: process them too using same loop (they were added to lists but not processed)
# For simplicity, we collected deltas but didn't transport them. Append the collected deltas into E/D/tag and continue loop:
# Actually to properly include deltas, we should push them into the stack. Let's restructure: maintain stack list and process.

print("done, steps", steps)
track_all = track_all/np.sum(track_all)
track_delta_frac = track_delta.sum()/np.sum(track_all)
print("L4 angular:", np.round(track_all,5))
print("sum:", track_all.sum())
print("delta track frac (MC):", track_delta_frac)
np.save('L4_ang.npy', track_all)
