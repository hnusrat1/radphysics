
import numpy as np
re=2.8179403e-13; ne=3.3428e23; me=0.51099895
def sigma_KN(E):
    k=E/me
    t1=(1+k)/k**2*(2*(1+k)/(1+2*k)-np.log(1+2*k)/k)
    t2=np.log(1+2*k)/(2*k)-(1+3*k)/(1+2*k)**2
    return 2*np.pi*re**2*(t1+t2)
for E in [0.1,0.2,1.1732,1.3325,2.0]:
    print(E, ne*sigma_KN(E))
