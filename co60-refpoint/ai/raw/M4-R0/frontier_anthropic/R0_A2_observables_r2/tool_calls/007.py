
def K0_w(): return 2*np.pi*re**2*mec2*NeW   # MeV cm^2 * cm^-3 ... per (g/cm3) factor
def above(T,D,variant):
    # N * integral_D^{T/2} eps dsigma/deps  (MeV/cm)
    if D>=T/2: return 0.0
    f=lambda eps: eps*moller_dsde(T,eps,variant)
    pts=[x for x in [D*3,D*10,T/10] if D<x<T/2]
    v,_=quad(f,D,T/2,limit=500,points=pts or None)
    return NeW*v
# test shape: compare the tau-dependence of (ESTAR_Scol - N*int_{D}^{T/2}) against Bethe F-term
def Fminus(tau):
    b2=1-1/(1+tau)**2
    return 1-b2+(tau**2/8-(2*tau+1)*np.log(2))/(tau+1)**2
def bethe_part(tau):  # ln(tau^2(tau+2)/2) + F^- (I-independent, ignoring -2ln(I))
    return np.log(tau**2*(tau+2)/2)+Fminus(tau)
D=0.01
for T in [0.1,0.3,0.5,1.0,2.0]:
    tau=T/mec2
    row=es[np.argmin(abs(es[:,0]-T))]
    Scol_E=row[1]  # MeV cm2/g
    K0=2*np.pi*re**2*mec2*NeW
    # restricted (D) by difference
    Lres = Scol_E - above(T,D,1)
    print(T, round(Scol_E,4), round(above(T,D,1),4), round(above(T,D,2),4), 'Bethe-ish', round(Scol_E/K0*1e0,4), round(bethe_part(tau),4))
