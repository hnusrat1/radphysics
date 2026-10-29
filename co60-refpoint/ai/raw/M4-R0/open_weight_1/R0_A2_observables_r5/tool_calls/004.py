import numpy as np, time

m_e = 0.5109989461

def sample_compton(E, rng, K=10):
    E = np.asarray(E, dtype=np.float64)
    n = E.size
    alpha = E/m_e
    def p_mu(mu, a):
        d = 1.0 + a[:,None]*(1.0-mu)
        return ((1+mu*mu)/2.0 + (a[:,None]**2*(1-mu)**2)/(2.0*d)) / (d*d)
    mu_c = rng.uniform(-1,1,(n,K))
    u_c = rng.random((n,K))
    p = p_mu(mu_c, alpha)
    accept = u_c < p
    has = accept.any(axis=1)
    idx = np.argmax(accept, axis=1)
    mu = mu_c[np.arange(n), idx]
    bad = ~has
    cnt=0
    while bad.any() and cnt<50:
        nb = bad.sum()
        mu_c2 = rng.uniform(-1,1,(nb,K)); u_c2=rng.random((nb,K))
        p2 = p_mu(mu_c2, alpha[bad])
        acc2 = u_c2 < p2
        h2 = acc2.any(axis=1)
        mu[bad] = mu_c2[np.arange(nb), np.argmax(acc2,axis=1)]
        bad[bad] = ~h2
        cnt+=1
    if bad.any():
        mu[bad]=rng.uniform(-1,1,bad.sum())
    Ep = E/(1.0 + alpha*(1.0-mu))
    pe = np.sqrt(np.maximum(E*E + Ep*Ep - 2*E*Ep*mu,0))
    cos_e = (E - Ep*mu)/pe
    Te = E - Ep
    return mu, Ep, cos_e, Te

rng = np.random.default_rng(1)
E = np.full(2000000, 1.25)
t0=time.time()
mu, Ep, cos_e, Te = sample_compton(E, rng)
print('compton sampling time', time.time()-t0, 'mean Te', Te.mean(), 'mean cos_e', cos_e.mean(), 'mean Ep', Ep.mean())