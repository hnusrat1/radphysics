
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
Eanch = np.array([0.05,0.1,0.2,0.3,0.5,1.0,1.25,1.5,2.0,3.0])
uanch = np.array([0.2269,0.1707,0.1370,0.1187,0.0969,0.0707,0.0632,0.0577,0.0494,0.0397])
def mu_tot(E):
    return np.exp(np.interp(np.log(E), np.log(Eanch), np.log(uanch)))

rng = np.random.default_rng(1)
N = 200000
E0 = 1.1732
z = rng.uniform(0,30,N)
rmax = 5.0*(1.0+z/100.0)
x = rng.uniform(-rmax, rmax, N)
y = rng.uniform(-rmax, rmax, N)
dV = (2*rmax)**2*30.0
sx=x; sy=y; sz=z+100.0
s1 = np.sqrt(sx*sx+sy*sy+sz*sz)
ux=sx/s1; uy=sy/s1; uz=sz/s1
wp1 = s1*z/(100.0+z)
wx=-x; wy=-y; wz=5.0-z
s2 = np.sqrt(wx*wx+wy*wy+wz*wz)
wx/=s2; wy/=s2; wz/=s2
csc = ux*wx+uy*wy+uz*wz
Ep = E0/(1.0+(E0/me)*(1.0-csc))
mup = mu_tot(Ep)
m0 = mu_tot(E0)
J = 0.5*(105.0/s1)**2*np.exp(-m0*wp1)
ds = kn_dsigma_dmu(E0,csc)
g = J*n_e_g*ds*np.exp(-mup*s2)/(s2*s2)
print("mean J",J.mean(),"mean ds",ds.mean(),"mean 1/s2^2",(1/s2**2).mean(),"mean dV",dV.mean(),"mean exp",np.exp(-mup*s2).mean())
print("mean g*n_e? ", (J*n_e_g*ds).mean(), "mean g", g.mean())
print("integral approx", (g*dV).sum())
print("check n_e_g*ds mean:", (n_e_g*ds).mean(), "(cm^-1)")
print("s2 mean", s2.mean(), "s2^2 mean", (s2**2).mean())
# simple estimate of volume
Vtot = 4*25*np.trapezoid((1+np.linspace(0,30,1001)/100)**2, np.linspace(0,30,1001))
print("Vtot", Vtot)
