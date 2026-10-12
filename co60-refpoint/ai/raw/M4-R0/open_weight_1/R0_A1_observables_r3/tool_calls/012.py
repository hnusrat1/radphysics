
import numpy as np
# direct integration of primary estimator over the field
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
N=2000000
dx = rng.uniform(-5,5,size=N); dy = rng.uniform(-5,5,size=N)
vec = np.stack([dx,dy,np.full(N,100.0)],axis=1)
u = vec/np.linalg.norm(vec,axis=1,keepdims=True)
pos = np.zeros((N,3)); pos[:,2]=-100.0
phi0ph = 1/(0.01*105**2)
for E in [1.1732,1.3325]:
    c = ray_cylinder_contrib(pos,u,np.full(N,E))
    print(f"line {E}: prim/phi0 = {c.sum()/N/phi0ph:.5f}")
c1 = ray_cylinder_contrib(pos,u,np.full(N,1.1732)); c2 = ray_cylinder_contrib(pos,u,np.full(N,1.3325))
print("avg:", (c1.sum()+c2.sum())/(2*N*phi0ph))
print("central-ray analytic:",0.5*(np.exp(-5*mu_tot(1.1732))+np.exp(-5*mu_tot(1.3325))))
