
import numpy as np
from scipy import integrate
np.random.seed(1)
re=2.8179403e-13; me=0.51099895
NA=6.02214076e23
# water: H 0.111894, O 0.888106 (mass fractions)
wH,wO=0.111894,0.888106
ZA_w=wH*1/1.00794+wO*8/15.9994   # electrons per gram / NA
ne_w=NA*ZA_w
ZA_air=0.49919
ne_air=NA*ZA_air
print("ne water",ne_w)

def kn_sigma(E):  # per electron, cm2, E in MeV
    k=E/me
    t1=(1+k)/k**2*(2*(1+k)/(1+2*k)-np.log(1+2*k)/k)
    t2=np.log(1+2*k)/(2*k)
    t3=(1+3*k)/(1+2*k)**2
    return 2*np.pi*re**2*(t1+t2-t3)
def mu_KN(E): return ne_w*kn_sigma(E)
def mu_tr_KN(E):
    # energy-transfer cross section per electron, integrate over angle
    def f(c):
        eps=1/(1+E/me*(1-c))
        dsig=0.5*re**2*eps**2*(eps+1/eps-(1-c**2))  # dsigma/dOmega per electron
        T=E*(1-eps)
        return dsig*2*np.pi*T
    val,_=integrate.quad(lambda c:f(c),-1,1)
    return ne_w*val
for E in [0.05,0.1,0.3,0.662,1.1732,1.25,1.3325]:
    print(E, mu_KN(E), mu_tr_KN(E)/ 1.0, mu_tr_KN(E)/mu_KN(E))
