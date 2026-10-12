
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

Eanch = np.array([0.05,0.1,0.2,0.3,0.5,1.0,1.25,1.5,2.0,3.0])
uanch = np.array([0.2269,0.1707,0.1370,0.1187,0.0969,0.0707,0.0632,0.0577,0.0494,0.0397])
def mu_tot(E):
    return np.exp(np.interp(np.log(np.clip(E,0.05,3.0)), np.log(Eanch), np.log(uanch)))

photon_bins = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
ct_bins = np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])

def run_singlescatter(E0, f, N, seed):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0,30,N)
    rmax = 5.0*(1.0+z/100.0)
    x = rng.uniform(-rmax, rmax, N)
    y = rng.uniform(-rmax, rmax, N)
    dV = (2*rmax)**2*30.0
    s1 = np.sqrt(x*x+y*y+(z+100.0)**2)
    ux = x/s1; uy = y/s1; uz = (z+100.0)/s1
    wp1 = s1*z/(100.0+z)
    sx = -x; sy = -y; sz = 5.0-z
    s2 = np.sqrt(sx*sx+sy*sy+sz*sz)
    ok = s2>0.25
    wx = sx/s2; wy=sy/s2; wz=sz/s2
    J = f*(105.0/s1)**2*np.exp(-mu_tot(E0)*wp1)
    csc = ux*wx+uy*wy+uz*wz
    eta = E0/me
    Ep = E0/(1.0+eta*(1.0-csc))
    mup = mu_tot(Ep)
    dsdmu = kn_dsigma_dmu(E0, csc)
    dsdom = dsdmu/(2*np.pi)   # d sigma / d Omega
    g = J*n_e_g*dsdom*np.exp(-mup*s2)/(s2*s2)
    I = (g*dV).mean()
    spec = np.histogram(Ep, bins=photon_bins, weights=g*dV)[0]/(N*I)
    ang  = np.histogram(wz, bins=ct_bins, weights=g*dV)[0]/(N*I)
    enflu = (g*dV*Ep).mean()
    return dict(I=I, spec=spec, ang=ang, enflu=enflu, Ep=Ep, wz=wz)

N=12_000_000
r1 = run_singlescatter(1.1732, 0.5, N, 101)
r2 = run_singlescatter(1.3325, 0.5, N, 202)
I_tot = r1['I']+r2['I']
en_tot = r1['enflu']+r2['enflu']
print("TOTAL single-scatter fluence/Phi0:", I_tot)
print("TOTAL single-scatter energy fluence/Phi0 (MeV):", en_tot)
print("mean scattered photon energy:", en_tot/I_tot)
print("primary fluence/Phi0:", 0.5*np.exp(-mu_tot(1.1732)*5)+0.5*np.exp(-mu_tot(1.3325)*5))
spec = r1['spec']+r2['spec']
ang = r1['ang']+r2['ang']
print("single-scatter spectrum (per Phi0, integral over bins = I):")
print(np.round(spec,5))
print("sum spec:", spec.sum())
print("single-scatter angular fractions (sum=1):")
print(np.round(ang,5), "sum", ang.sum())
