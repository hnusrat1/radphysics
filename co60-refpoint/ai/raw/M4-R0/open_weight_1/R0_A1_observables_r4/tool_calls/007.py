
import numpy as np, time

me = 0.51099895
re = 2.8179403262e-13
NA = 6.02214076e23
n_e_g = NA*10/18.01528

def kn_total(E):
    eta = E/me
    a = 1.0+2.0*eta
    return 2*np.pi*re**2*((1+eta)/eta**2*(2*(1+eta)/a - np.log(a)/eta) + np.log(a)/(2*eta) - (1+3*eta)/a**2)

# cross sections (per g water)
Egrid = np.array([0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.1732,1.25,1.3325,1.5,2.0,3.0])
muR  = np.array([0.028,0.012,0.0085,0.0060,0.0052,0.0045,0.0033,0.0028,0.0022,0.0019,0.0017,0.0015,0.0012,0.0010,0.0009,0.00085,0.0008,0.00075,0.00055,0.00040])
tau  = np.array([0.60,0.15,0.045,0.0093,0.0026,0.0019,0.00070,0.00042,0.00016,0.000085,0.000052,0.000036,0.000021,0.000013,0.000010,0.000009,0.000008,0.000007,0.000004,0.0000015])
kap  = np.zeros_like(Egrid); kap[Egrid>1.022]=np.interp(np.clip(Egrid[Egrid>1.022],1.1732,3.0),[1.1732,1.3325,1.5,2.0,3.0],[0.00012,0.00075,0.0011,0.0026,0.0055])

def interp1(x, xp, fp):
    return np.interp(np.clip(x, xp[0], xp[-1]), xp, fp)

def mu_comp(E): return kn_total(np.clip(E,0.01,3.0))*n_e_g
def mu_ray(E): return interp1(E, Egrid, muR)
def mu_photo(E): return interp1(E, Egrid, tau)
def mu_pair(E): return np.where(E>2*me, interp1(E, Egrid, kap), 0.0)
def mu_totX(E):
    return mu_comp(E)+mu_ray(E)+mu_photo(E)+mu_pair(E)

for E in [0.1,0.5,1.1732,1.3325]:
    print(E, "mu_c",round(mu_comp(E),5),"mu_R",round(mu_ray(E),5),"mu_ph",round(mu_photo(E),6),"mu_p",round(mu_pair(E),6),"tot",round(mu_totX(E),5))

# ---- KN inverse CDF table ----
eta_grid = np.linspace(0.02, 3.0, 400)   # eta = E/me
xi_grid = np.linspace(0.001, 0.999, 512)
mu_grid = np.linspace(-1, 1, 4001)
# CDF as function of mu for each eta
# dsigma/dmu = pi re^2 r^2 (r + 1/r - sin^2), r=1/(1+eta(1-mu))
def kn_dsdmu_vec(eta, mu):
    r = 1.0/(1.0+eta*(1.0-mu))
    return np.pi*re**2 * r**2 * (r + 1.0/r - (1.0-mu*mu))

Q = np.zeros((len(eta_grid), len(xi_grid)))
for i,eta in enumerate(eta_grid):
    d = kn_dsdmu_vec(eta, mu_grid)
    c = np.cumsum(d)
    c = c/c[-1]
    Q[i,:] = np.interp(xi_grid, c, mu_grid)

def sample_kn_cos(E, xi):
    eta = np.clip(E/me, 0.02, 3.0)
    # linear interpolation over eta grid
    ie = (eta - eta_grid[0])/(eta_grid[-1]-eta_grid[0])*(len(eta_grid)-1)
    i0 = np.clip(np.floor(ie).astype(int),0,len(eta_grid)-2)
    f = ie-i0
    q0 = Q[i0,:]; q1 = Q[i0+1,:]
    # interpolate in xi (uniform grid)
    ix = xi*(len(xi_grid)-1)
    j0 = np.clip(np.floor(ix).astype(int),0,len(xi_grid)-2)
    fx = ix-j0
    v0 = q0[j0]*(1-fx)+q0[j0+1]*fx
    v1 = q1[j0]*(1-fx)+q1[j0+1]*fx
    return v0*(1-f)+v1*f

# test sampling validity
xi = np.random.default_rng(0).random(200000)
mu = sample_kn_cos(np.full(200000,1.1732), xi)
print("sampled mu mean, hist check:",mu.mean(), np.histogram(mu,bins=10,range=(-1,1))[0]/200000)

# rotate direction
def rotate(u, cosT, phi):
    # rotate unit vector u by polar angle theta (cosT) around random axis defined by phi
    # construct orthonormal basis: e1 perpendicular to u
    ux,uy,uz = u[...,0],u[...,1],u[...,2]
    nz = np.abs(ux)<0.9
    ex = np.where(nz, 1.0,0.0)
    # e1 = cross(e_zhat, u) normalized; e2 = cross(u,e1)
    # e1 = (uy, -ux, 0) normalized for general u? cross(zhat,u) = (uy,-ux,0)
    e1x = uy; e1y = -ux; e1z = np.zeros_like(ux)
    n = np.sqrt(e1x*e1x+e1y*e1y)
    e1x/=n; e1y/=n
    e2x = uy*e1z-uz*e1y; e2y = uz*e1x-ux*e1z; e2z = ux*e1y-uy*e1x
    st = np.sqrt(np.clip(1-cosT*cosT,0,1))
    cphi = np.cos(phi); sphi = np.sin(phi)
    nx = e1x*(st*cphi)+e2x*(st*sphi)+ux*cosT
    ny = e1y*(st*cphi)+e2y*(st*sphi)+uy*cosT
    nz2 = e1z*(st*cphi)+e2z*(st*sphi)+uz*cosT
    return np.stack([nx,ny,nz2],axis=-1)
