
import numpy as np
re=2.8179403e-13; mc2=0.51099895; ne=3.3428e23  # electrons/cm3 water (rho=1)
def sigKN(E):
    k=E/mc2
    a=1+k
    t1=(1+k)/k**2*(2*(1+k)/(1+2*k)-np.log(1+2*k)/k)
    t2=np.log(1+2*k)/(2*k)
    t3=(1+3*k)/(1+2*k)**2
    return 2*np.pi*re**2*(t1+t2-t3)
def muKN(E): return ne*sigKN(E)
# NIST water total mu/rho (cm2/g), from memory: 
nist={1.0:0.07072,1.25:0.06327,1.5:0.05646}
for E in [0.5,1.0,1.1732,1.25,1.3325,1.5,2.0]:
    print(E, round(muKN(E),5), nist.get(E))
