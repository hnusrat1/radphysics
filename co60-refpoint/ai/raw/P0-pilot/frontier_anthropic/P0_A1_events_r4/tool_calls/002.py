
def fbind(E): return 1-0.046*np.clip(E/2.0,1e-3,None)**0.42
def mu_inc(E): return sKN(E)*ne*fbind(E)
def mu_pair(E):
    f=np.clip((E-1.022)/0.978,0,None); return 2.66e-3*f**1.5
def mu_tot(E): return mu_inc(E)+mu_pe(E)+mu_coh(E)+mu_pair(E)
print(mu_tot(2.0), mu_tot(0.1), mu_tot(1.0), mu_tot(0.5))
mugrid=np.log(mu_tot(Egrid))
