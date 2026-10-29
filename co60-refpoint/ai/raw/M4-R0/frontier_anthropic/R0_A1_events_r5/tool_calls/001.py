
import numpy as np, time
rng = np.random.default_rng(12345)
me = 0.51099895
NA_e = 3.3428e23          # electrons per gram (water, Z/A=0.55509)
N_mol = 3.3428e22         # water molecules per cm^3
r_e = 2.8179403e-13       # cm
MU_PAIR_CM = 0.0001       # cm^-1 (rough; negligible)
MU_PE_CM   = 0.0          # cm^-1 (negligible at MeV)

def kn_sigma(E):
    k = E/me
    a = (1+k)/k**2
    return 2*np.pi*r_e**2*(2*a*(2*k*(1+k)/(1+2*k)-np.log(1+2*k))/k*0 + 0) if False else \
        2*np.pi*r_e**2*((1+k)/k**2*(2*(1+k)/(1+2*k)-np.log(1+2*k)/k) + np.log(1+2*k)/(2*k) - (1+3*k)/(1+2*k)**2)

def rayleigh_sigma(lam_A):
    a = 0.3
    # sigma = r_e^2 * int (1+cos^2)/2 * F^2 dOmega ; F = 10/(1+(s/a)^2)^2, s = sin(th/2)/lambda
    th = np.linspace(1e-9, np.pi, 2000001)
    s = np.sin(th/2)/lam_A
    F2 = 100/(1+(s/a)**2)**4
    f = 0.5*(1+np.cos(th)**2)*F2*2*np.pi*np.sin(th)
    return r_e**2*np.trapz(f, th)

energies = np.array([1.1732, 1.3325])
mu = {}
for E in energies:
    sig_inc = kn_sigma(E)
    lam = 12.398/(E*1000)
    sig_coh = rayleigh_sigma(lam)
    mu_inc = N_mol*0 + NA_e*sig_inc  # per cm (rho=1 => per gram*1)
    mu_coh = N_mol*sig_coh
    mu[E] = dict(inc=mu_inc, coh=mu_coh, pair=MU_PAIR_CM, pe=MU_PE_CM)
    print(E, mu[E], "total", sum(mu[E].values()))
