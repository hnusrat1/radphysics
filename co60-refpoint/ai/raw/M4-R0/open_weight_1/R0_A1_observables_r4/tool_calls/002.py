
import numpy as np

me = 0.51099895
re = 2.8179403262e-13
NA = 6.02214076e23
n_e_g = NA*10/18.01528

def kn_dsigma_dmu(E, mu):
    eta = E/me
    r = 1.0/(1.0+eta*(1.0-mu))
    sint2 = 1.0-mu*mu
    return np.pi*re**2 * r**2 * (r + 1.0/r - sint2)

def kn_total(E):
    eta = E/me
    a = 1.0+2.0*eta
    return 2*np.pi*re**2*((1+eta)/eta**2*(2*(1+eta)/a - np.log(a)/eta) + np.log(a)/(2*eta) - (1+3*eta)/a**2)

# NIST-like total attenuation for water (log-log interpolated anchors)
Eanch = np.array([0.05,0.1,0.2,0.3,0.5,1.0,1.25,1.5,2.0,3.0])
uanch = np.array([0.2269,0.1707,0.1370,0.1187,0.0969,0.0707,0.0632,0.0577,0.0494,0.0397])
def mu_tot(E):
    return np.exp(np.interp(np.log(E), np.log(Eanch), np.log(uanch)))

# energy-absorption
mench = np.array([0.05,0.1,0.2,0.3,0.5,1.0,1.25,1.5,2.0])
menv  = np.array([0.0410,0.0255,0.0364,0.0342,0.0330,0.0310,0.0297,0.0283,0.0261])
def mu_en(E):
    return np.exp(np.interp(np.log(E), np.log(mench), np.log(menv)))

print("mu_tot 1.1732,1.3325:",mu_tot(1.1732),mu_tot(1.3325))
print("mu_en 1.1732,1.3325:",mu_en(1.1732),mu_en(1.3325))

# single scatter Monte Carlo
rng = np.random.default_rng(12345)
N = 8_000_000
lines = [(1.1732,0.5),(1.3325,0.5)]
P = np.array([0.0,0.0,5.0])

photon_bins = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
ct_bins = np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])

def run_singlescatter(E0, f, N, seed):
    rng = np.random.default_rng(seed)
    # sample z uniform [0,30], x,y uniform in [-5(1+z/100), 5(1+z/100)]
    z = rng.uniform(0,30,N)
    rmax = 5.0*(1.0+z/100.0)
    x = rng.uniform(-rmax, rmax, N)
    y = rng.uniform(-rmax, rmax, N)
    dV = (2*rmax)**2 * 30.0  # per-sample volume weight
    # incident direction
    sx = x; sy = y; sz = z+100.0
    s1 = np.sqrt(sx*sx+sy*sy+sz*sz)
    ux = sx/s1; uy = sy/s1; uz = sz/s1
    # water path to scatter point
    wp1 = s1*z/(100.0+z)
    # direction to P
    wx = -x; wy = -y; wz = 5.0-z
    s2 = np.sqrt(wx*wx+wy*wy+wz*wz)
    wx/=s2; wy/=s2; wz/=s2
    mu0 = mu_tot(np.full(N,E0))
    mu_s = mu_tot(E0)  # placeholder
    w0 = np.full(N, E0)
    mup = mu_tot(E0)
    # scattered photon energy (Compton)
    csc = ux*wx+uy*wy+uz*wz
    eta = E0/me
    Ep = E0/(1.0+eta*(1.0-csc))
    mup = mu_tot(Ep)
    dsdmu = kn_dsigma_dmu(E0, csc)
    # incident fluence per Phi0: f*(105/s1)^2*exp(-mu0*wp1)
    J = f*(105.0/s1)**2*np.exp(-mu0*wp1)
    contrib = J * n_e_g * dsdmu * np.exp(-mup*s2)/(s2*s2) * dV
    # exclude points within 0.3 cm of P (handled separately, tiny)
    mask = s2 > 0.3
    contrib = contrib[mask]
    Ep = Ep[mask]; wz = wz[mask]
    tot = contrib.sum()
    # spectrum
    spec = np.histogram(Ep, bins=photon_bins, weights=contrib)[0]
    ang = np.histogram(wz, bins=ct_bins, weights=contrib)[0]
    enflu = (contrib*Ep).sum()
    return tot, spec, ang, enflu, contrib, Ep

res = {}
for E0,f in lines:
    tot, spec, ang, enflu, c, Ep = run_singlescatter(E0, f, N, seed=int(E0*1000))
    res[E0] = dict(tot=tot, spec=spec, ang=ang, enflu=enflu)
    print(f"E0={E0}: single-scatter fluence/Phi0 = {tot:.5f}, energy fluence/Phi0*MeV = {enflu:.5f}")

totSS = res[1.1732]['tot']+res[1.3325]['tot']
enSS = res[1.1732]['enflu']+res[1.3325]['enflu']
print("Total single-scatter fluence/Phi0:", totSS)
print("mean scattered energy:", enSS/totSS)
print("primary fluence:", 0.5*np.exp(-mu_tot(1.1732)*5)+0.5*np.exp(-mu_tot(1.3325)*5))
