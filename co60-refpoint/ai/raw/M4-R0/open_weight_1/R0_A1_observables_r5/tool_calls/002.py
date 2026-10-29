
import numpy as np

# Klein-Nishina total cross section per electron (barn), gamma = E/mc2
r_e = 2.8179403262e-13  # cm
def sigma_KN(E_MeV):
    g = E_MeV/0.51099895
    a = 1+2*g
    s = 2*np.pi*r_e**2 * ( (1+g)/g**2 * (2*(1+g)/a - np.log(a)/g) + np.log(a)/(2*g) - (1+3*g)/a**2 )
    return s/(1e-24)  # barn

# electrons per gram water: 10 electrons / 18.01528 g * Na
Na = 6.02214076e23
e_per_g_water = 10.0/18.01528*Na  # per gram
e_per_g_air = 14.462/28.964*Na*?  # N2 (14 e, 28) + O2(16,32) + Ar(18,40)
# air: 0.755 N2 + 0.232 O2 + 0.013 Ar by mass
e_per_g_air = (0.75524*14/28.0134 + 0.23199*16/31.998 + 0.01277*18/39.948)*Na

for E in [0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.5,0.8,1.0,1.1732,1.25,1.3325,1.5,2.0]:
    print(E, sigma_KN(E), sigma_KN(E)*e_per_g_water/1e24)
