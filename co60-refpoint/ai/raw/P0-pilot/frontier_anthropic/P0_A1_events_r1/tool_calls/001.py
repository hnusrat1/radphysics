
import numpy as np
re=2.8179403e-13; me=0.510998950
def sigKN(E):  # cm^2 per electron, E in MeV
    k=E/me
    t1=(1+k)/k**2*(2*(1+k)/(1+2*k)-np.log(1+2*k)/k)
    t2=np.log(1+2*k)/(2*k)-(1+3*k)/(1+2*k)**2
    return 2*np.pi*re**2*(t1+t2)
Ne=6.02214076e23*(10/18)*1.0  # electrons per gram*density: water 3.343e23 e/g
Ne=0.5551*6.02214076e23
print(sigKN(2.0)*Ne, sigKN(1.0)*Ne)
