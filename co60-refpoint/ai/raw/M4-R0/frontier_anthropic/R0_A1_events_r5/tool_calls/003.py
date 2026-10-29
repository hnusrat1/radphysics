
import numpy as np
rng = np.random.default_rng(12345)
me = 0.51099895
NA_e = 3.3428e23
N_mol = 3.3428e22
r_e = 2.8179403e-13
MU_PAIR_CM = 0.0001
MU_PE_CM = 0.0
energies = np.array([1.1732, 1.3325])
mu = {}
for E in energies:
    sig_inc = kn_sigma(E)
    lam = 12.398/(E*1000)
    sig_coh = rayleigh_sigma(lam)
    mu[E] = dict(inc=NA_e*sig_inc, coh=N_mol*sig_coh, pair=MU_PAIR_CM, pe=MU_PE_CM)
    print(E, {k:round(v,6) for k,v in mu[E].items()}, "total", sum(mu[E].values()))
