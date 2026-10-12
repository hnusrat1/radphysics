
import sys; sys.path.insert(0,'/tmp')
import numpy as np
from scipy.integrate import quad
import work_base as wb
es=wb.es; mec2=wb.mec2; re=2.8179403e-13
def beta2(T): 
    g=1+T/mec2; return 1-1/g**2
def moller_dsde(T,eps,variant):
    tau=T/mec2; x=eps/T
    b2=beta2(T)
    c = (2*tau+1)/(tau+2)**2 if variant==1 else (2*tau+1)/(tau+1)**2
    B=1/x**2+1/(1-x)**2+(tau/(tau+2))**2-c/(x*(1-x))
    return 2*np.pi*re**2*mec2/(b2*T**2)*B
NeW=6.02214076e23*0.55509
def Scol(T,variant,I=75e-6):
    f=lambda eps: eps*moller_dsde(T,eps,variant)
    val,err=quad(f,I,T/2,limit=500,points=[1e-3,1e-2] if T>0.02 else None)
    return NeW*val
for T in [0.1,0.5,1.0,2.0]:
    row=es[np.argmin(abs(es[:,0]-T))]
    print(T, row[0], Scol(T,1), Scol(T,2), 'ESTAR', row[1])
