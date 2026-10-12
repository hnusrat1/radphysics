import numpy as np

mec2 = 0.5110
N_e_per_g = 6.02214076e23*10/18.01528
n_e = N_e_per_g
C = 0.4987e-24*0.511
X0_water = 36.08  # cm

def moler_dsde(T, E0):
    if T<=0.011 or E0<=0.010 or E0>=T/2: return 0.0
    beta2 = 1.0-1.0/(1+T/mec2)**2
    if beta2<=0: return 0.0
    return C*n_e/beta2*(1/E0**2 - 1/(E0*(T-E0)) + 1/(T-E0)**2)

def Scol_w(E):
    return np.interp(E, [0.01,0.02,0.03,0.05,0.07,0.1,0.2,0.3,0.5,0.7,1.0,1.25,1.4],
                     [22.7,11.3,7.97,6.36,5.14,4.098,2.63,2.16,1.94,1.89,1.847,1.847,1.85])
def Srest_w(E):
    if E<=0.011: return Scol_w(E)
    dEint=0.00025
    E0v=np.arange(0.0105, E/2-1e-9, dEint)
    return Scol_w(E) - np.sum(E0v*np.array([moler_dsde(E,x) for x in E0v])*dEint)
def pbeta(E):
    return max(E*(E+2*mec2)/(E+mec2), 1e-6)

rng = np.random.default_rng(11)

# load q and L3 direction
d = np.load('L4_data.npz')
Eg = d['Eg']; q = d['q']
dE = 0.002
# initial direction distribution from L3 (all): stored earlier
L3 = np.load('L3_data.npy', allow_pickle=True).item()
ang_tot = L3['ang_tot']; ang_tot_n = ang_tot/ang_tot.sum()
ct_bins = np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
ct_c = (ct_bins[:-1]+ct_bins[1:])/2

N_e_elec = 300000
# sample initial energies from q
pw = q*q  # weight proportional to q (number of electrons), use q directly as prob density
pw = np.maximum(q,0)
prob = pw/pw.sum()
E0s = rng.choice(Eg, size=N_e_elec, p=prob)
# initial directions
cos0 = rng.choice(ct_c, size=N_e_elec, p=ang_tot_n)
az0 = rng.uniform(0,2*np.pi,size=N_e_elec)

def dir_from(cosT, az):
    s = np.sqrt(np.maximum(1-cosT**2,0))
    return np.array([s*np.cos(az), s*np.sin(az), cosT])

dx = np.zeros(N_e_elec); dy = np.zeros(N_e_elec); dz = np.zeros(N_e_elec)
dx, dy, dz = dir_from(cos0, az0)
# state arrays; we'll process in a growable list: use arrays and append deltas
E = E0s.copy()
tag = np.zeros(N_e_elec)  # 0 primary-origin, 1 delta-origin
track_all = np.zeros(10)
track_delta = np.zeros(10)
alive = np.ones(N_e_elec,bool)
ds = 0.01
def step_electrons(E_, dx_,dy_,dz_, tag_, nsteps):
    # transport one step for arrays; returns updated E,dirs; track tally; also delta creations appended
    # local losses
    Sres = np.array([Srest_w(e) for e in E_])
    dE_loc = Sres*ds
    E_ = E_ - dE_loc
    # multiple scattering
    pv = np.array([pbeta(e) for e in np.maximum(E_,0.011)])
    sig = (14.1/pv)*np.sqrt(ds/X0_water)*(1+0.038*np.log(ds/X0_water))
    th = np.abs(rng.normal(0,sig))
    az = rng.uniform(0,2*np.pi,size=len(E_))
    # rotate direction by th around axis
    cosT = np.cos(th)
    sinT = np.sin(th)
    # new dir: rotate (dx,dy,dz) by th with azimuth az: use Rodrigues-like in terms of spherical: easier: sample new direction with polar angle th around old dir
    # random axis perpendicular to old dir:
    # n = normalize(cross(zhat?,dir)) - simpler approach: use a rotation matrix per particle is heavy; use trick:
    # newdir = cosT*dir + sinT*(cos az * m + sin az * n) where m,n orthonormal to dir
    # build n: cross(dir, random vector)
    rv = rng.normal(size=(len(E_),3))
    dirs = np.stack([dx_,dy_,dz_],axis=1)
    ndot = np.sum(rv*dirs,axis=1,keepdims=True)
    n = rv - ndot*dirs
    nn = np.linalg.norm(n,axis=1,keepdims=True)+1e-12
    n = n/nn
    m = np.cross(dirs,n)
    caz = np.cos(az); saz = np.sin(az)
    newd = cosT[:,None]*dirs + sinT[:,None]*(caz[:,None]*m + saz[:,None]*n)
    newd = newd/np.linalg.norm(newd,axis=1,keepdims=True)
    # tally track in cos bin (we use the new direction as representative; fine)
    ctz = np.clip(newd[:,2],-1,1)
    bi = np.clip(np.searchsorted(ct_bins,ctz,side='right')-1,0,9)
    np.add.at(track_all, bi, ds)
    np.add.at(track_delta, bi, ds*tag_)
    # delta production
    E_new = E_.copy()
    extra_E = []; extra_dx=[]; extra_dy=[]; extra_dz=[]; extra_tag=[]
    for i in range(len(E_)):
        T = E_[i]
        if T < 0.012: continue
        # prob of delta in step: ∫_Δ^{T/2} dΣ dE0 * ds
        E0v = np.arange(0.0105, T/2-1e-9, 0.0005)
        if len(E0v)==0: continue
        dS = np.array([moler_dsde(T,x) for x in E0v])
        pdel = dS.sum()*0.0005*ds
        if rng.uniform()<min(pdel,1):
            # sample E0 from distribution
            w = dS/dS.sum()
            E0 = rng.choice(E0v, p=w)
            # primary loses E0 (jump)
            E_new[i] -= E0
            # secondary direction: forward-ish
            cosd = np.sqrt(E0/(E0+2*mec2))
            azd = rng.uniform(0,2*np.pi)
            extra_E.append(E0); extra_tag.append(1.0)
            # secondary direction relative to primary's new direction: cos from primary dir
            # approximate: secondary at angle with cosd from primary direction, random azimuth
            sd,cd2,sad = np.sin(np.arccos(np.clip(cosd,-1,1))), cosd, np.sin(0)
            # axis perpendicular to primary dir
            dirp = newd[i]
            rv = rng.normal(size=3)
            nvec = rv - np.dot(rv,dirp)*dirp
            nvec = nvec/(np.linalg.norm(nvec)+1e-12)
            mvec = np.cross(dirp,nvec)
            c2,s2 = np.cos(azd), np.sin(azd)
            dnew = cosd*dirp + np.sin(np.arccos(np.clip(cosd,-1,1)))*(c2*mvec + s2*nvec)
            dnew = dnew/(np.linalg.norm(dnew)+1e-12)
            extra_dx.append(dnew[0]); extra_dy.append(dnew[1]); extra_dz.append(dnew[2])
    # return updated arrays along with extra electrons to append later
    return E_new, newd[:,0], newd[:,1], newd[:,2], extra_E, extra_dx, extra_dy, extra_dz, extra_tag

# process all electrons including deltas with a simple stack approach
# we'll use arrays and a loop over steps; deltas appended into lists and processed afterwards
stack_E = list(E); stack_d = list(zip(dx.tolist(),dy.tolist(),dz.tolist())); stack_tag = list(tag)
processed = 0
while processed < len(stack_E):
    # take a batch of current electrons
    batch_E = np.array(stack_E[processed:])
    batch_d = np.array(stack_d[processed:])
    batch_tag = np.array(stack_tag[processed:])
    n_b = len(batch_E)
    E_b = batch_E; dx_b=batch_d[:,0]; dy_b=batch_d[:,1]; dz_b=batch_d[:,2]; tag_b=batch_tag
    # transport until all die
    while n_b>0:
        # step
        res = step_electrons(E_b, dx_b,dy_b,dz_b, tag_b, 1)
        E_b, dx_b,dy_b,dz_b, exE, exdx,exdy,exdz, extag = res
        # append extras to stack
        for k in range(len(exE)):
            stack_E.append(exE[k]); stack_d.append((exdx[k],exdy[k],exdz[k])); stack_tag.append(extag[k])
        alive_ = E_b > 0.0105
        E_b = E_b[alive_]; dx_b=dx_b[alive_]; dy_b=dy_b[alive_]; dz_b=dz_b[alive_]; tag_b=tag_b[alive_]
        n_b = len(E_b)
    processed += n_b_orig if False else 0
    # easier: mark processed by original batch length
print("done track MC")

track_all = track_all/np.sum(track_all)
track_delta_frac = track_delta.sum()/(track_all.sum())
print("L4 angular (fraction):", np.round(track_all,5))
print("sum:", track_all.sum())
print("delta fraction of track from MC:", track_delta_frac)
np.save('L4_ang.npy', track_all)
