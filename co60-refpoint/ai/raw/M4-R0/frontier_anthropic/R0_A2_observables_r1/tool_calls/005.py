
import sys, os
sys.path.insert(0,'/tmp/work')
import lib, numpy as np
print(lib.Om_p, lib.mutot(np.array([1.1732])), lib.ne*0)
# check KN sampler & sigma
mu=np.linspace(-1,1,20001)
E=1.1732
sig=2*np.pi*lib.re**2/2*np.trapz(lib.kn_f(E,mu),mu)
print("n_e*sigKN =",lib.ne*sig, " XCOM inc =", lib.mu_inc(np.array([E])))
rng=np.random.default_rng(1)
m=lib.kn_sample_mu(np.full(200000,E),rng)
print("mean mu sampled",m.mean(), "analytic", np.trapz(mu*lib.kn_f(E,mu),mu)/np.trapz(lib.kn_f(E,mu),mu))
