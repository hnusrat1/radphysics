import numpy as np, math
mec2 = 0.51099895
r_e = 2.8179403262e-13
Na  = 6.02214076e23
molW = 18.01528
e_per_g_water = 10.0/molW*Na
mol_per_g_water = Na/molW
e_per_g_air = (0.75524*14/28.0134 + 0.23199*16/31.998 + 0.01277*18/39.948)*Na
I_water = 78e-6
I_air   = 85.7e-6
ZA_water = 10.0/18.01528
ZA_air = 0.4992

def KN_barn(E):
    g = E/mec2
    a = 1+2*g
    return 2*np.pi*r_e**2*((1+g)/g**2*(2*(1+g)/a-np.log(a)/g)+np.log(a)/(2*g)-(1+3*g)/a**2)/1e-24

def mu_inc_water(E):
    E=np.asarray(E,float)
    return KN_barn(E)*e_per_g_water/1e24

def mu_coh_water(E):
    return 0.00085*(1.25/np.asarray(E,float))**0.65

def mu_photo_water(E):
    return 1.2e-5*np.asarray(E,float)**-3.0

def mu_pair_water(E):
    E=np.asarray(E,float)
    out=np.zeros_like(E)
    m=E>1.022
    x=(E[m]-1.022)/9.0
    out[m]=(0.114+2*0.0018)*x*x*1e-24*mol_per_g_water
    return out

def mu_tot_water(E):
    return mu_inc_water(E)+mu_coh_water(E)+mu_photo_water(E)+mu_pair_water(E)

Kc = 0.153536
def s_col_formula(E_MeV, ZA, I_MeV):
    E=np.asarray(E_MeV,float)
    tau=E/mec2
    gam=tau+1.0
    beta2=1.0-1.0/gam**2
    Tmax = tau/2.0*mec2
    num = 2*mec2*beta2*gam**2*Tmax
    B = np.log(num/I_MeV**2) - beta2 - (2*gam-1)/gam**2*np.log(2.0) + (gam-1)**2/(8*gam**2)
    return Kc*ZA/beta2*B

for E in [0.01,0.02,0.05,0.1,0.2,0.5,1.0,1.25,2.0]:
    print(E, s_col_formula(E,ZA_water,I_water))
