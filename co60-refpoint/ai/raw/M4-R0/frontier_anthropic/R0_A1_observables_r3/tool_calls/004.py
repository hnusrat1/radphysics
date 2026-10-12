
# radiative yield g(T)
Tg=np.logspace(-3,np.log10(1.5),600)
def Sratio(T): return rad_S(T)/(rad_S(T)+Scol_bethe(T,Iw,neg_w))
gT=np.zeros_like(Tg)
for i,T in enumerate(Tg):
    x=np.logspace(-4,np.log10(T),400) if T>1e-4 else np.array([1e-5,1e-4])
    I=np.trapezoid(Sratio(x),x)
    gT[i]=I/T
def g_of_T(T): return np.interp(np.log(np.maximum(T,1e-3)),np.log(Tg),gT)
# KN tables
xg,wg=np.polynomial.legendre.leggauss(300)
def KN_table(E):
    k=E/me; eps=1/(1+k*(1-xg))
    f=0.5*re**2*eps**2*(eps+1/eps-(1-xg**2))
    T=E*(1-eps)
    s=2*np.pi*np.sum(wg*f)
    Tm=np.sum(wg*f*T)/np.sum(wg*f)
    TgT=np.sum(wg*f*T*(1-g_of_T(T)))/np.sum(wg*f)
    return s,Tm,TgT
Etab=np.logspace(-3,np.log10(1.5),400)
tab=np.array([KN_table(E) for E in Etab])
def interpE(arr,E): return np.interp(np.log(np.maximum(E,1e-3)),np.log(Etab),arr)
Tmean_tab=tab[:,1]; TgT_tab=tab[:,2]
print(KN_table(1.1732), KN_sigma(1.1732))
