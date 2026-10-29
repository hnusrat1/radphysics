
import numpy as np, time
re=2.8179403e-13; ne=3.3428e23; mec2=0.51099895
Om_p_int = integrate.dblquad(lambda v,u:(1+u*u+v*v)**-1.5, -0.05,0.05, -0.05,0.05)[0]
print("Omega_p (sr)",Om_p_int)
def kn_f(E,mu):
    k=E/mec2; eps=1/(1+k*(1-mu))
    return eps*eps*(eps+1/eps-(1-mu*mu))   # dsig/dOmega = re^2/2 * f
def sigKN(E):
    mug=np.linspace(-1,1,4001)
    f=kn_f(E,mug[None,:]) if np.ndim(E) else kn_f(E,mug)
    return 2*np.pi*re**2/2*np.trapz(f,mug,axis=-1)
print("sigKN(1.1732) barn/e", sigKN(np.array([1.1732]))/1e-24, " n_e*sig", ne*sigKN(np.array([1.1732]))[0], mutot(np.array([1.1732]))*0+lli(Ex,X[:,2],np.array([1.1732])))
