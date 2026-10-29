
import sys; sys.path.insert(0,'/tmp/work')
import numpy as np, lib
trapz=np.trapezoid
mu=np.linspace(-1,1,20001); E=1.1732
sig=2*np.pi*lib.re**2/2*trapz(lib.kn_f(E,mu),mu)
print("n_e*sigKN =",lib.ne*sig, " XCOM inc =", lib.mu_inc(np.array([E])))
rng=np.random.default_rng(1)
m=lib.kn_sample_mu(np.full(200000,E),rng)
print("mean mu sampled",m.mean(), "analytic", trapz(mu*lib.kn_f(E,mu),mu)/trapz(lib.kn_f(E,mu),mu))
S=lib.S
mec2=lib.mec2
def moller_dsig(Tp,eps):
    tau=Tp/mec2; b2=tau*(tau+2)/(tau+1)**2
    br=1/eps**2+1/(Tp-eps)**2-(2*tau+1)/((tau+1)**2*eps*(Tp-eps))+(tau/(tau+1))**2/Tp**2
    return 2*np.pi*lib.re**2*mec2/b2*br
for Tp in [0.1,0.5,1.0,1.25]:
    ee=np.linspace(75e-6,Tp/2,400001)
    Sd=lib.ne*trapz(ee*moller_dsig(Tp,ee),ee)
    Scol=np.interp(Tp,S[:,0],S[:,1])
    ee2=np.linspace(0.01,Tp/2,40001)
    Sd10=lib.ne*trapz(ee2*moller_dsig(Tp,ee2),ee2)
    print(Tp, "delta-part all:",Sd," Scol ESTAR:",Scol, " >10keV part:",Sd10, "L_Delta approx", Scol-Sd10)
