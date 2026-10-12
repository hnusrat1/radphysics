
from scipy import integrate
def moller_f(x,tau):
    return 1/x**2+1/(1-x)**2+(tau/(tau+1))**2-(2*tau+1)/((tau+1)**2*x*(1-x))
def dsig_deps(T,eps):
    tau=T/me; b2=beta2(T)
    x=eps/T
    return 2*np.pi*re**2*me/(b2*T**2)*moller_f(x,tau)
for T in [0.1,0.5,1.0,1.3]:
    Ic=75e-6
    # energy loss to delta rays with eps>Ic (approx cutoff) vs Bethe
    val,_=integrate.quad(lambda e: e*dsig_deps(T,e)*ne,Ic,T/2,limit=200,points=[1e-3,1e-2])
    print(T, val, S_col_bethe(T), S_col_bethe(T)-val)
