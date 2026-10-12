import numpy as np

me=0.51099895; re=2.8179403262e-13; Ne=3.343e23
def kn_f(mu,E):
    a=E/me; r=1.0/(1.0+a*(1.0-mu))
    return r*r*(1.0/r+r-(1.0-mu*mu))

for E in [0.02,0.05,0.1,0.2,0.5,1.0,2.0]:
    mus=np.linspace(-1,1,3001)
    sig=np.pi*re**2*np.trapezoid(kn_f(mus,E),mus)
    mu_c=Ne*sig
    # compute photo using our table
    E_ph=np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.15,0.20,0.30,0.50,0.80,1.0,1.5,2.0])
    mu_ph=np.array([4.90,0.506,0.162,0.0712,0.0375,0.0223,0.00988,0.00547,0.00240,0.00128,0.000526,0.000151,5e-5,3e-5,1e-5,4.7e-6])
    logE=np.log(np.clip(E,0.01,2.0))
    mu_ph_e=np.exp(np.interp(logE,np.log(E_ph),np.log(mu_ph)))
    E_co=np.array([0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20,0.30,0.50,0.80,1.0,1.5,2.0])
    mu_co=np.array([0.30,0.060,0.022,0.0098,0.0046,0.0034,0.0017,0.0011,0.0006,0.0003,0.00015,0.00012,8e-5,4.5e-5])
    mu_co_e=np.exp(np.interp(logE,np.log(E_co),np.log(mu_co)))
    mu_tot=mu_c+mu_ph_e+mu_co_e
    print(f"E={E:.2f}: mu_c={mu_c:.5f}, mu_ph={mu_ph_e:.5f}, mu_co={mu_co_e:.5f}, tot={mu_tot:.5f}, ph_frac={mu_ph_e/mu_tot:.4f}, c_frac={mu_c/mu_tot:.4f}, co_frac={mu_co_e/mu_tot:.4f}")