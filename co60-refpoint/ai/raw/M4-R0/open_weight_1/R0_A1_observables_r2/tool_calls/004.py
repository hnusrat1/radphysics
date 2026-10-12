
import numpy as np

mec2=0.5109989461
re2=(2.8179403262e-13)**2

def moller_dsdT(E,T):
    gamma=1.0+E/mec2
    beta2=1.0-1.0/(gamma*gamma)
    C=2*np.pi*re2*mec2/beta2
    bracket = 1.0/T**2 - (2*gamma-1)/(gamma*gamma*T*(E-T)) + 1.0/(E-T)**2
    return C*bracket

for E in [0.1,0.5,1.0,1.33]:
    gamma=1.0+E/mec2
    beta2=1.0-1.0/(gamma*gamma)
    I=78e-6
    Tmax=E/2
    grid=np.geomspace(I, Tmax, 1000000)
    dT=np.diff(grid)
    T=(grid[1:]+grid[:-1])/2
    vals=T*moller_dsdT(E,T)
    integ=np.sum(vals*dT)
    Z_A=10/18.01528
    S_bethe=0.153536*Z_A*(np.log(2*mec2*beta2*gamma**2*Tmax/(I*I))-beta2)/beta2
    print(f"E={E}: integral T*dsdT = {integ:.3f}, S_bethe = {S_bethe:.3f} MeV cm2/g")
