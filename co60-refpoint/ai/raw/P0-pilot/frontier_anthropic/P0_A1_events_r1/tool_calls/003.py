
import numpy as np
from scipy.integrate import quad
re=2.8179403e-13; me=0.510998950
def sigKN(E):
    k=E/me
    t1=(1+k)/k**2*(2*(1+k)/(1+2*k)-np.log(1+2*k)/k)
    t2=np.log(1+2*k)/(2*k)-(1+3*k)/(1+2*k)**2
    return 2*np.pi*re**2*(t1+t2)
def shape(eps,k):
    c=1-(1/eps-1)/k
    return 1/eps+eps-(1-c*c)
def sig_from_shape(E):
    k=E/me; emin=1/(1+2*k)
    val,_=quad(lambda e: shape(e,k),emin,1,limit=200)
    return np.pi*re**2/k**2*val
print(sig_from_shape(2.0), sigKN(2.0), sig_from_shape(0.1), sigKN(0.1))
