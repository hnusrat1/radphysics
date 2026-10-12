
import numpy as np, pickle

with open('photon_mc.pkl','rb') as f:
    data = pickle.load(f)
res = data['res']; z0s = data['z0s']; N = data['N']
phi0ph = 1/(0.01*105**2)

# photon bins
ph_bins = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
ph_c = (ph_bins[:-1]+ph_bins[1:])/2
i5 = np.where(z0s==5.0)[0][0]
prim_spec5 = res['prim_spec'][i5]/(N*phi0ph)
scat_spec5 = res['scat_spec'][i5]/(N*phi0ph)
print("prim spec sum:", prim_spec5.sum(), "scat spec sum:", scat_spec5.sum())
print("prim flu/phi0 (MC):", res['prim_flu'][i5]/(N*phi0ph), "scat:", res['scat_flu'][i5]/(N*phi0ph))

# precise primary via direct integration
# recompute quickly
mec2=0.5110
N_e_per_g = 6.02214076e23*10/18.01528
E_tab = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0])
mu_tab = np.array([4.951,1.374,0.7031,0.4470,0.2269,0.2059,0.1837,0.1706,0.1505,0.1370,0.1186,0.1061,0.0969,0.0896,0.0786,0.0707,0.0636,0.0576,0.0494])
def interp(x,xe,ye): return np.interp(x,xe,ye)
def mu_tot(E): return interp(E,E_tab,mu_tab)
def ray_cylinder_contrib(p, u, E, z0=5.0, R=1.0, h=0.1, V=0.6283185):
    N = len(p)
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
rng = np.random.default_rng(5)
Ns=3000000
dx = rng.uniform(-5,5,size=Ns); dy = rng.uniform(-5,5,size=Ns)
vec = np.stack([dx,dy,np.full(Ns,100.0)],axis=1)
u = vec/np.linalg.norm(vec,axis=1,keepdims=True)
pos = np.zeros((Ns,3)); pos[:,2]=-100.0
c1 = ray_cylinder_contrib(pos,u,np.full(Ns,1.1732))
c2 = ray_cylinder_contrib(pos,u,np.full(Ns,1.3325))
prim_direct = (c1.sum()+c2.sum())/(2*Ns*phi0ph)
prim_direct_l1 = c1.sum()/(Ns*phi0ph); prim_direct_l2 = c2.sum()/(Ns*phi0ph)
print("primary/phi0 direct:", prim_direct, " per line: l1", prim_direct_l1, " l2", prim_direct_l2)

prim_flu5 = prim_direct
scat_flu5 = res['scat_flu'][i5]/(N*phi0ph)
tot_flu5 = prim_flu5+scat_flu5
print("scat_flu5:", scat_flu5, "total:", tot_flu5, "scatter frac:", scat_flu5/tot_flu5)

# primary spectrum: put direct per-line fluences into bins
prim_spec = np.zeros(28)
prim_spec[23] = 0.5*prim_direct_l1
prim_spec[26] = 0.5*prim_direct_l2
print("prim spec:", prim_spec, "sum:", prim_spec.sum())
scat_spec = scat_spec5

# energy-weighted
E_ph = np.sum((prim_spec+scat_spec)*ph_c)/(prim_spec.sum()+scat_spec.sum())
E_scat = np.sum(scat_spec*ph_c)/scat_spec.sum() if scat_spec.sum()>0 else 0
print("mean photon energy:", E_ph, " mean scat energy:", E_scat)
print("scattered energy-fluence fraction:", np.sum(scat_spec*ph_c)/np.sum((prim_spec+scat_spec)*ph_c))

# angular
prim_ang = res['prim_ang']/(N*phi0ph)
scat_ang = res['scat_ang']/(N*phi0ph)
print("scat ang/phi0:", scat_ang, "sum:", scat_ang.sum())
print("prim ang/phi0:", prim_ang, "sum:", prim_ang.sum())

# Save derived basic L1 numbers for later
np.save('prim_spec.npy', prim_spec); np.save('scat_spec.npy', scat_spec)
np.save('prim_ang.npy', prim_ang); np.save('scat_ang.npy', scat_ang)
print("saved")
