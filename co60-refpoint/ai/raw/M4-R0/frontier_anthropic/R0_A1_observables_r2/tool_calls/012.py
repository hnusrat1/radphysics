
import numpy as np
re=2.8179403e-13; mc2=0.51099895; NA=6.02214076e23
def Scol(T, I_eV, ne_per_g):
    tau=np.asarray(T)/mc2; g=tau+1; b2=1-1/g**2
    Kg=2*np.pi*re**2*mc2*ne_per_g
    Im=I_eV*1e-6/mc2
    F=1-b2+(tau**2/8-(2*tau+1)*np.log(2))/g**2
    return Kg/b2*(np.log(tau**2*(tau+2)/(2*Im**2))+F)
ne_w=0.55509*NA; ne_a=0.49919*NA
T=np.array([0.1,0.5,1.0,2.0])
print("water Scol", Scol(T,75.0,ne_w))
print("ESTAR water collision ref ~ 0.1:4.115  1:1.849")
print("air Scol", Scol(T,85.7,ne_a))
