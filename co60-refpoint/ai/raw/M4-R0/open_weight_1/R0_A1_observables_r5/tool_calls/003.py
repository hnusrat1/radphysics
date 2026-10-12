
import numpy as np

r_e = 2.8179403262e-13  # cm
def sigma_KN_barn(E_MeV):
    g = E_MeV/0.51099895
    a = 1+2*g
    s = 2*np.pi*r_e**2 * ( (1+g)/g**2 * (2*(1+g)/a - np.log(a)/g) + np.log(a)/(2*g) - (1+3*g)/a**2 )
    return s/1e-24  # barn

Na = 6.02214076e23
e_per_g_water = 10.0/18.01528*Na
e_per_g_air = (0.75524*14/28.0134 + 0.23199*16/31.998 + 0.01277*18/39.948)*Na

print("e/g water:", e_per_g_water/1e23, " air:", e_per_g_air/1e23)
print("E(MeV)  sigma_KN(b)  incoherent_cm2/g_water")
for E in [0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.5,0.8,1.0,1.1732,1.25,1.3325,1.5,2.0]:
    print(f"{E:.4g} {sigma_KN_barn(E):.4g} {sigma_KN_barn(E)*e_per_g_water/1e24:.6g}")
