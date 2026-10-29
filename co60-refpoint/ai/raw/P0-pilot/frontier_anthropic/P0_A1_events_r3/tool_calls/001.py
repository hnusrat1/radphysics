
import numpy as np
re=2.8179403e-13; me=0.51099895
NA=6.02214076e23
ne=1.0*(2*1+8)/18.015*NA   # electrons per gram of water
def sigKN(E):
    k=E/me; L=np.log(1+2*k)
    return 2*np.pi*re**2*((1+k)/k**2*(2*(1+k)/(1+2*k)-L/k)+L/(2*k)-(1+3*k)/(1+2*k)**2)
for E in [0.1,0.2,0.511,1.0,2.0]:
    print(E, ne*sigKN(E))
print("residual at 2 MeV (total 0.04942):", 0.04942-ne*sigKN(2.0))
