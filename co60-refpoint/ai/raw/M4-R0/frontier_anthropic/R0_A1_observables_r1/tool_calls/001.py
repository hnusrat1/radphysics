
import numpy as np
re=2.8179403e-13; mc2=0.51099895; ne=3.3428e23
def sigKN(E):
    e=E/mc2; L=np.log(1+2*e)
    return 2*np.pi*re**2*((1+e)/e**2*(2*(1+e)/(1+2*e)-L/e)+L/(2*e)-(1+3*e)/(1+2*e)**2)
def fKN(E,c):  # per sr, per electron
    Ep=1/(1+E/mc2*(1-c)); s2=1-c*c
    return re**2/2*Ep**2*(Ep+1/Ep-s2)
# checks
c=np.linspace(-1,1,200001)
for E in [0.1,0.5,1.1732,1.3325]:
    num=2*np.pi*np.trapz(fKN(E,c),c)
    print(E, num, sigKN(E), ne*sigKN(E))
# max of f/(re^2/2)
print(np.max(fKN(1.0,c)/(re**2/2)))
