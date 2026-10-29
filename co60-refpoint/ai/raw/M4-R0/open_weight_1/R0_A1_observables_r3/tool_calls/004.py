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

def kn_diff(cosT, E):
    al = E/mec2
    x = 1+al*(1-cosT)
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
    new = new/np.linalg.norm(new,axis=1,keepdims=True)
    return new

def ray_cylinder_contrib(p, u, E, z0, R=1.0, h=0.1, V=0.6283185):
    N = len(p)
    if N==0:
        return np.zeros(0), np.zeros(0,bool)
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
    return cont/V, hit

if __name__ == '__main__':
    rng = np.random.default_rng(1)
    Etest = np.full(200000,1.25)
    Ep, xt, cp = sample_compton(Etest,rng)
    print("check KN: mean Ee", (Etest-Ep).mean(), " (expect ~0.64-0.68)")
    
    print("mu totals:", mu_tot(1.1732), mu_tot(1.3325), "prim/phi0:", 0.5*(np.exp(-5*mu_tot(1.1732))+np.exp(-5*mu_tot(1.3325))))
