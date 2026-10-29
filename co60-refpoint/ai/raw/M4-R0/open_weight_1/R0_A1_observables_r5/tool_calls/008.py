import numpy as np
mec2 = 0.51099895
Kc = 0.153536
I_water = 78e-6
I_air   = 85.7e-6
ZA_water = 10.0/18.01528
ZA_air = 0.4992

def s_col_formula(E_MeV, ZA, I_MeV):
    E=np.asarray(E_MeV,float)
    tau=E/mec2
    gam=tau+1.0
    beta2=1.0-1.0/gam**2
    Tmax = tau/2.0*mec2
    num = 2*mec2*beta2*gam**2*Tmax
    B = np.log(num/I_MeV**2) - beta2 - (2*gam-1)/gam**2*np.log(2.0) + (gam-1)**2/(8*gam**2)
    return Kc*ZA/beta2*B

def l_delta_formula(E_MeV, ZA, I_MeV, delta):
    # restricted stopping power with cutoff delta (MeV), transfers <= delta
    E=np.asarray(E_MeV,float)
    tau=E/mec2
    gam=tau+1.0
    beta2=1.0-1.0/gam**2
    num = 2*mec2*beta2*gam**2*delta
    # exchange correction evaluated with delta in place of Tmax approx:
    B = np.log(num/I_MeV**2) - beta2 - (2*gam-1)/gam**2*np.log(2.0) + (gam-1)**2/(8*gam**2)
    return Kc*ZA/beta2*B

def s_rad_water(E):
    E=np.asarray(E,float)
    Z2A=3.667
    # calibrate: S_rad(1 MeV)=0.0073
    A = 0.0073/1.511
    return A*Z2A*(E+mec2)

for E in [0.1,0.5,1.0,1.25,2.0]:
    sw=s_col_formula(E,ZA_water,I_water)
    sa=s_col_formula(E,ZA_air,I_air)
    print(E, 'water',round(sw,4),'air',round(sa,4),'ratio',round(sw/sa,4),'Lw',round(l_delta_formula(E,ZA_water,I_water,0.01),4),'La',round(l_delta_formula(E,ZA_air,I_air,0.01),4),'Lratio',round(l_delta_formula(E,ZA_water,I_water,0.01)/l_delta_formula(E,ZA_air,I_air,0.01),4))
print('s_rad water',[round(s_rad_water(E),4) for E in [0.1,0.5,1.0,1.25,2.0]])
