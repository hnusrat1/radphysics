import numpy as np, scipy, math, time

# ---------- constants ----------
mec2 = 0.51099895   # MeV
r_e = 2.8179403262e-13  # cm
Na  = 6.02214076e23
molW = 18.01528        # water g/mol
e_per_g_water = 10.0/molW*Na
mol_per_g_water = Na/molW
e_per_g_air = (0.75524*14/28.0134 + 0.23199*16/31.998 + 0.01277*18/39.948)*Na
I_water = 78e-6       # MeV
I_air   = 85.7e-6     # MeV
ZA_water = 10.0/18.01528
ZA_air = 0.4992

def KN_barn(E):
    g = E/mec2
    a = 1+2*g
    return 2*np.pi*r_e**2*((1+g)/g**2*(2*(1+g)/a-np.log(a)/g)+np.log(a)/(2*g)-(1+3*g)/a**2)/1e-24

def mu_inc_water(E):   # cm^2/g
    E=np.asarray(E,float)
    return KN_barn(E)*e_per_g_water/1e24

def mu_coh_water(E):
    E=np.asarray(E,float)
    return 0.00085*(1.25/E)**0.65

def mu_photo_water(E):
    E=np.asarray(E,float)
    return 1.2e-5*E**-3.0

def mu_pair_water(E):
    E=np.asarray(E,float)
    out=np.zeros_like(E)
    m=E>1.022
    x=(E[m]-1.022)/9.0
    sO=0.114*x*x*1e-24
    sH=0.0018*x*x*1e-24
    out[m]=(sO+2*sH)*mol_per_g_water
    return out

def mu_tot_water(E):
    return mu_inc_water(E)+mu_coh_water(E)+mu_photo_water(E)+mu_pair_water(E)

def mu_processes(E):
    inc=mu_inc_water(E); coh=mu_coh_water(E); pho=mu_photo_water(E); pai=mu_pair_water(E)
    tot=inc+coh+pho+pai
    return inc/tot, coh/tot, pho/tot, pai/tot

# electron stopping power: calibrate Bethe formula
Kc = 0.153536  # MeV cm^2/g
def s_col_formula(E_MeV, ZA, I_MeV):
    E=np.asarray(E_MeV,float)
    tau=E/mec2
    gam=tau+1.0
    beta2=1.0-1.0/gam**2
    Tmax = tau/2.0*mec2
    num = 2*mec2*beta2*gam**2*Tmax
    B = np.log(num/I_MeV**2) - beta2 - (2*gam-1)/gam**2*np.log(2.0) + (gam-1)**2/(8*gam**2)
    return Kc*ZA/beta2*B

Egrid=np.array([0.01,0.02,0.05,0.1,0.2,0.5,1.0,1.25,2.0,5.0,10.0])
Sw_est=np.array([22.4,11.6,5.42,4.10,2.53,1.98,1.852,1.84,1.84,1.86,1.88])  # ESTAR-like water collision
Sa_est=Sw_est*(0.4992/0.5551)*?  # placeholder

# test formula vs estimate
for E in [0.1,0.5,1.0,1.25,2.0]:
    print(E, s_col_formula(E,ZA_water,I_water))
print('ratios at 1 MeV:', s_col_formula(1.0,ZA_water,I_water), s_col_formula(1.0,ZA_air,I_air))
