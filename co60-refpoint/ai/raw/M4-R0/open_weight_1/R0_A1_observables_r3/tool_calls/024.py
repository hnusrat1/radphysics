
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
def moler_dsde(T, E0):
    if T<=0.011 or E0<=0.010 or E0>=T/2: return 0.0
    beta2 = 1.0-1.0/(1+T/mec2)**2
    if beta2<=0: return 0.0
    return C*n_e/beta2*(1/E0**2 - 1/(E0*(T-E0)) + 1/(T-E0)**2)
Eg_all = np.arange(0.010,1.401,0.002)
Srest_grid = np.array([Scol_w(np.array([e]))[0] - (np.sum(np.arange(0.0105, e/2-1e-9, 0.00025)*np.array([moler_dsde(e,x) for x in np.arange(0.0105, e/2-1e-9, 0.00025)])*0.00025) if e>0.011 else 0.0) for e in Eg_all])
Scol_grid = np.array([Scol_w(e) for e in Eg_all])
def Srest_fast(E): return np.interp(E, Eg_all, Srest_grid)
def Scol_fast(E): return np.interp(E, Eg_all, Scol_grid)
def pbeta(E): return np.maximum(E*(E+2*mec2)/(E+mec2), 1e-6)

d = np.load('L4_data.npz')
Eg = d['Eg']; q = d['q']
L3 = np.load('L3_data.npy', allow_pickle=True).item()
ang_tot_n = L3['ang_tot']/L3['ang_tot'].sum()
ct_bins = np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])

rng = np.random.default_rng(11)
N_elec = 200000
prob = np.maximum(q,0); prob = prob/prob.sum()
E0s = rng.choice(Eg, size=N_elec, p=prob)
cos0 = rng.choice((ct_bins[:-1]+ct_bins[1:])/2, size=N_elec, p=ang_tot_n)
az0 = rng.uniform(0,2*np.pi,size=N_elec)
s0 = np.sqrt(np.maximum(1-cos0**2,0))
D0 = np.stack([s0*np.cos(az0), s0*np.sin(az0), cos0],axis=1)

track_all = np.zeros(10); track_delta = np.zeros(10)
ds = 0.02
t0=time.time()

def tally(Di, tag_i, w):
    ctz = np.clip(Di[:,2],-1,1)
    bi = np.clip(np.searchsorted(ct_bins,ctz,side='right')-1,0,9)
    np.add.at(track_all, bi, w)
    np.add.at(track_delta, bi, w*tag_i)

jobs = [(E0s, D0, np.zeros(N_elec))]
n_jobs_processed = 0
total_steps = 0
while jobs:
    E, D, tag = jobs.pop(0)
    m = len(E)
    while m > 0:
        # one step
        Sres = Srest_fast(E)
        E = E - Sres*ds
        pv = pbeta(np.maximum(E,0.0112))
        sig = (14.1/pv)*np.sqrt(ds/X0_water)*(1+0.038*np.log(ds/X0_water))
        th = np.abs(rng.normal(0,sig,size=m))
        az = rng.uniform(0,2*np.pi,size=m)
        cosT = np.cos(th); sinT = np.sin(th)
        rv = rng.normal(size=(m,3))
        ndot = np.sum(rv*D,axis=1,keepdims=True)
        nvec = (rv - ndot*D)/(np.linalg.norm(rv-ndot*D,axis=1,keepdims=True)+1e-12)
        mvec = np.cross(D,nvec)
        caz = np.cos(az); saz = np.sin(az)
        D = cosT[:,None]*D + sinT[:,None]*(caz[:,None]*mvec + saz[:,None]*nvec)
        D = D/np.linalg.norm(D,axis=1,keepdims=True)
        tally(D, tag, ds)
        # delta production
        beta2 = 1.0 - 1.0/(1+E/mec2)**2
        okE = E > 0.022
        pdel = np.zeros(m)
        T = E[okE]; bet2 = beta2[okE]
        pdel[okE] = ds*(C*n_e/bet2)*(1.0/DELTA + np.log((T-DELTA)/DELTA)/T - 1.0/(T-DELTA))
        pdel = np.minimum(pdel, 1.0)
        creates = rng.uniform(size=m) < pdel
        if creates.any():
            Tc = E[creates]
            nc = creates.sum()
            U = rng.uniform(1e-9,1-1e-9,size=nc)
            E0 = 1.0/(1.0/DELTA - U*(1.0/DELTA - 2.0/Tc))
            r = E0/(Tc-E0)
            acc = rng.uniform(size=nc) < (1 - r + r*r)
            while not acc.all():
                rem = ~acc
                U2 = rng.uniform(1e-9,1-1e-9,size=rem.sum())
                E0[rem] = 1.0/(1.0/DELTA - U2*(1.0/DELTA - 2.0/Tc[rem]))
                r2 = E0[rem]/(Tc[rem]-E0[rem])
                acc[rem] = rng.uniform(size=rem.sum()) < (1-r2+r2*r2)
            E[creates] -= E0
            cosd = np.sqrt(np.maximum(E0/(E0+2*mec2),1e-9))
            ad = np.arccos(np.clip(cosd,-1,1))
            azd = rng.uniform(0,2*np.pi,size=nc)
            dirp = D[creates]
            rv2 = rng.normal(size=(nc,3))
            nd2 = np.sum(rv2*dirp,axis=1,keepdims=True)
            nv2 = (rv2 - nd2*dirp)/(np.linalg.norm(rv2-nd2*dirp,axis=1,keepdims=True)+1e-12)
            mv2 = np.cross(dirp,nv2)
            c2v = np.cos(azd); s2v = np.sin(azd)
            dd = np.cos(ad)[:,None]*dirp + np.sin(ad)[:,None]*(c2v[:,None]*mv2 + s2v[:,None]*nv2)
            dd = dd/np.linalg.norm(dd,axis=1,keepdims=True)
            jobs.append((E0, dd, np.ones(nc)))
        alive = E > 0.0105
        E = E[alive]; D = D[alive]; tag = tag[alive]
        m = len(E)
        total_steps += 1
        if total_steps % 5000 == 0:
            print(f"  steps {total_steps}, alive {m}, jobs {len(jobs)}, elapsed {time.time()-t0:.0f}s", flush=True)
        if total_steps > 200000: break
    if total_steps > 200000: break

print("finished, total steps", total_steps)
track_all_n = track_all/np.sum(track_all)
print("L4 angular fractions:", np.round(track_all_n,5))
print("sum:", track_all_n.sum())
print("delta track fraction MC:", track_delta.sum()/(track_all.sum()))
np.save('L4_ang.npy', track_all_n)
