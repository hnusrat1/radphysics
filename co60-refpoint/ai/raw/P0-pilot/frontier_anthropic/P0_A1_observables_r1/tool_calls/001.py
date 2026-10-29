
import numpy as np
re=2.8179403e-13; mc2=0.51099895; ne=3.3428e23
def sig_KN(E):
    k=E/mc2
    L=np.log(1+2*k)
    return 2*np.pi*re**2*((1+k)/k**2*(2*(1+k)/(1+2*k)-L/k)+L/(2*k)-(1+3*k)/(1+2*k)**2)
E=2.0
print("KN per e (cm2)",sig_KN(E), " mu_c/rho =",ne*sig_KN(E))
print("max scattered energy-grid issue: Compton E' range", E/(1+2*E/mc2), E)
