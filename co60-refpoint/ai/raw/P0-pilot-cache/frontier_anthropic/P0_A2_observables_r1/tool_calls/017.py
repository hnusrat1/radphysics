
import numpy as np, sys; sys.path.insert(0,'/tmp/work'); import xs
RE=2.8179403e-13; ME=0.51099895; NA=6.02214076e23
K0=2*np.pi*RE**2*ME*NA   # MeV cm2/mol-ish: times (Z/A)/beta2
def berger_S(T,ZA,I_eV):
    tau=T/ME; b2=1-1/(1+tau)**2
    Im=I_eV*1e-6/ME
    Fm=(1-b2)*(1+tau**2/8-(2*tau+1)*np.log(2))
    return K0*ZA/b2*(np.log(tau**2*(tau+2)/(2*Im**2))+Fm)
for T in [0.01,0.05,0.1,0.5,1.0,2.0,5.0]:
    print(T, berger_S(T,0.55509,75.0), xs.Scol(T))
