
def moller_dsig(T, eps):
    # dsigma/deps per electron (cm2/MeV), T kinetic of primary, eps delta energy (<= T/2)
    b2=beta2(T); tau=T/me
    x=eps/T
    f=1/x**2+1/(1-x)**2+(tau/(tau+1))**2-(2*tau+1)/((tau+1)**2*x*(1-x))
    return 2*np.pi*re**2*me/b2/T**2*f
def Ldelta(T, Dcut=0.01):
    # int_{Dcut}^{T/2} eps dsig/deps d eps  (cm2 MeV/electron)
    out=np.zeros_like(T)
    for i,t in enumerate(np.atleast_1d(T)):
        if t/2<=Dcut: continue
        e=np.exp(np.linspace(np.log(Dcut),np.log(t/2),400))
        out[i]=np.trapezoid(e*moller_dsig(t,e),e)
    return out
def Sres_mass(T,mat,Dcut=0.01):
    # restricted (delta < Dcut) collision stopping power per gram
    Sc=Scol_per_mass(T,mat)
    Ld=Ldelta(T,Dcut)*(mat['ne'])  # MeV/cm per electron-density
    return Sc*1.0 - Ld
print('check Sres water 0.1', Sres_mass(np.array([0.1]),WATER), 'Ldelta(0.1)*ne', Ldelta(np.array([0.1]))*WATER['ne'])
