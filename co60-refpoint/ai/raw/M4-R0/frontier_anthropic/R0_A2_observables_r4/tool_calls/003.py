
re=2.8179403e-13; mc2=0.51099895; ne=3.3428e23
def kn_total(E):  # cm^2 per electron, E MeV
    k=E/mc2
    L=np.log(1+2*k)
    return 2*np.pi*re**2*((1+k)/k**2*(2*(1+k)/(1+2*k)-L/k)+L/(2*k)-(1+3*k)/(1+2*k)**2)
for e in [0.1,0.5,1.1732,1.3325]:
    print(e, kn_total(e)*ne, mu_inc(e))
# Moller check of collision stopping power, T in MeV
def moller_dsig(T,Td):  # dsigma/dTd, cm^2/MeV, Td<=T/2
    tau=T/mc2; g=1+tau; b2=1-1/g**2
    eps=Td/T
    F=1/eps**2+1/(1-eps)**2+(tau/(tau+2))**2-(2*tau+1)/((tau+1)**2*eps*(1-eps))
    return 2*np.pi*re**2*mc2/(b2*T**2)*F
from scipy.integrate import quad
def Lrest(T,Dlo,Dhi):  # n_e * int T_d dsig over [Dlo, min(Dhi,T/2)] , MeV cm^2/g (rho=1 -> per cm)
    hi=min(Dhi,T/2)
    if hi<=Dlo: return 0.0
    return ne*quad(lambda td: td*moller_dsig(T,td),Dlo,hi,limit=200)[0]/ (1/1)  # per cm
for T in [0.1,0.5,1.0,2.0]:
    Ls=ne*quad(lambda td: td*moller_dsig(T,td),0.075e-3,T/2,limit=400)[0]
    Ldel=Lrest(T,0.01,1e9)
    # ESTAR collision (MeV cm2/g) interpolated
    Es=np.interp(T,E[:,0],E[:,1])
    print(T, Ls/ne*ne/ne*1, Ls, Es, Ldel, Ls/6.022e23*0+0)
