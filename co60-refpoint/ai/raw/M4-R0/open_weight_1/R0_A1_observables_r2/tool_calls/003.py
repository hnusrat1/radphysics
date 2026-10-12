
import numpy as np

mec2=0.5109989461
re2=(2.8179403262e-13)**2

def moller_dsdT(E,T):
    # T and E in MeV, gamma from kinetic E
    gamma=1.0+E/mec2
    beta2=1.0-1.0/(gamma*gamma)
    C=2*np.pi*re2*mec2/beta2
    bracket = 1.0/T**2 - (2*gamma-1)/(gamma*gamma*T*(E-T)) + 1.0/(E-T)**2
    return C*bracket

for E in [1.0]:
    gamma=1.0+E/mec2
    beta2=1.0-1.0/(gamma*gamma)
    I=78e-6
    Tmax=E/2
    # integrate T * dsigma
    grid=np.geomspace(I, Tmax, 200000)
    dT=np.diff(grid)
    T=(grid[1:]+grid[:-1])/2
    vals=T*moller_dsdT(E,T)
    S=n_e*0 # placeholder
    print("integral T*dsdT =", np.sum(vals*dT))
    # Bethe for water
    Z_A=10/18.01528
    S_bethe=0.153536*Z_A*(np.log(2*mec2*beta2*gamma**2*Tmax/(I*I))-beta2)/beta2
    print("S_bethe=", S_bethe)
