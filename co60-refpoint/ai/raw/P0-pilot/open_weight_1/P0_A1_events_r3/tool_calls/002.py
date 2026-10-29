import numpy as np

me = 0.51099895   # MeV
re = 2.8179403262e-13  # cm
Ne = 3.343e23     # electrons per cm^3 in water

# --- water attenuation tables (mass coeff cm^2/g, density 1 -> cm^-1) ---
# photoelectric
E_ph = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.15,0.20,0.30,0.50,0.80,1.0,1.5,2.0])
mu_ph= np.array([4.90,0.506,0.162,0.0712,0.0375,0.0223,0.00988,0.00547,0.00240,0.00128,0.000526,0.000151,5.0e-5,3.0e-5,1.0e-5,4.7e-6])
# pair production
E_pa = np.array([1.022,1.25,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0,10.0])
mu_pa= np.array([0.0,3.1e-5,1.32e-4,4.1e-4,8.1e-4,1.39e-3,2.69e-3,4.32e-3,6.19e-3,1.05e-2,1.55e-2])
# coherent (Rayleigh)
E_co = np.array([0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20,0.30,0.50,0.80,1.0,1.5,2.0])
mu_co= np.array([0.30,0.060,0.022,0.0098,0.0046,0.0034,0.0017,0.0011,0.0006,0.0003,0.00015,0.00012,8e-5,4.5e-5])

def kn_f(mu, E):
    a = E/me
    r = 1.0/(1.0+a*(1.0-mu))
    return r*r*(1.0/r + r - (1.0-mu*mu))

# check fmax
for E in [0.05,0.2,0.5,1.0,2.0]:
    mus = np.linspace(-1,1,200001)
    f = kn_f(mus,E)
    print(f"E={E}: fmax={f.max():.4f} at mu={mus[np.argmax(f)]:.4f}")

# KN total cross section per electron, precompute on grid
def kn_sigma(E):
    mus = np.linspace(-1,1,4001)
    f = kn_f(mus,E)
    return np.pi*re**2*np.trapz(f,mus)
Egrid = np.logspace(np.log10(0.001), np.log10(10.0), 400)
sig_grid = np.array([kn_sigma(E) for E in Egrid])
print("sigma_KN(2MeV)=",np.interp(np.log(2.0),np.log(Egrid),sig_grid), " -> mu_c=",Ne*np.interp(np.log(2.0),np.log(Egrid),sig_grid))
