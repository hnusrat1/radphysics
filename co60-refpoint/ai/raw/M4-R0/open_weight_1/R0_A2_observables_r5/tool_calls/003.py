import numpy as np, time, os, json

xcom = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
E_x = xcom[:,0]; mu_coh = xcom[:,1]; mu_inc = xcom[:,2]; mu_pe = xcom[:,3]
mu_pair = xcom[:,4]+xcom[:,5]; mu_tot = xcom[:,6]
LE = np.log(E_x)

def mut_at(E, mu):
    return np.exp(np.interp(np.log(np.clip(E,1e-6,20)), LE, np.log(mu)))

# test Compton KN sampling speed and correctness
m_e = 0.5109989461

def sample_compton(E, n, rng):
    # returns cos_theta (list per photon first accepted), E', electron angle cos
    alpha = E/m_e
    # p(mu)
    def p_mu(mu, a):
        d = 1.0 + a*(1.0-mu)
        return ((1+mu*mu)/2.0 + (a*a*(1-mu)**2)/(2.0*d)) / (d*d)
    K = 10
    mu_c = rng.uniform(-1,1,(n,K))
    u_c = rng.random((n,K))
    # pmax=1 at mu=1
    p = p_mu(mu_c, alpha)
    accept = u_c < p
    # first accepted per row
    has = accept.any(axis=1)
    idx = np.argmax(accept, axis=1)  # if none accepted returns 0
    mu = mu_c[np.arange(n), idx]
    # for rows with none accepted, resample via loop
    bad = ~has
    cnt=0
    while bad.any() and cnt < 100:
        nb = bad.sum()
        mu_c2 = rng.uniform(-1,1,(nb, K))
        u_c2 = rng.random((nb,K))
        p2 = p_mu(mu_c2, alpha[bad] if np.ndim(alpha)>0 else alpha)
        acc2 = u_c2 < p2
        h2 = acc2.any(axis=1)
        idx2 = np.argmax(acc2, axis=1)
        mu[bad] = mu_c2[np.arange(nb), idx2]
        bad[bad] = ~h2
        cnt+=1
    # fallback if still bad
    if bad.any():
        mu[bad] = rng.uniform(-1,1, bad.sum())
    # energy and electron direction
    Ep = E/(1.0 + alpha*(1.0-mu))
    pe = np.sqrt(E*E + Ep*Ep - 2*E*Ep*mu)
    cos_e = (E - Ep*mu)/pe
    Te = E - Ep
    return mu, Ep, cos_e, Te

rng = np.random.default_rng(1)
E = np.full(2000000, 1.25)
t0=time.time()
mu, Ep, cos_e, Te = sample_compton(E, len(E), rng)
print('compton sampling time', time.time()-t0, 'mean Te', Te.mean(), 'mean cos_e', cos_e.mean())
# analytic check: mean Te for 1.25 MeV ~ ?
print('mean Ep', Ep.mean())