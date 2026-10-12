import numpy as np
r_e = 2.8179403262e-13
Na = 6.02214076e23
e_per_g_water = 10.0/18.01528*Na
def KN_barn(E):
    g=E/0.51099895
    a=1+2*g
    return 2*np.pi*r_e**2*((1+g)/g**2*(2*(1+g)/a-np.log(a)/g)+np.log(a)/(2*g)-(1+3*g)/a**2)/1e-24
for E in [0.1,0.5,1,1.25,2,3,5,8,10,20]:
    print(E, KN_barn(E), KN_barn(E)*e_per_g_water/1e24)
