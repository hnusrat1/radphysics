
def kn_sigma(E):
    k = E/me
    return 2*np.pi*r_e**2*((1+k)/k**2*(2*(1+k)/(1+2*k)-np.log(1+2*k)/k) + np.log(1+2*k)/(2*k) - (1+3*k)/(1+2*k)**2)

def rayleigh_sigma(lam_A):
    a = 0.3
    th = np.linspace(1e-9, np.pi, 2000001)
    s = np.sin(th/2)/lam_A
    F2 = 100/(1+(s/a)**2)**4
    f = 0.5*(1+np.cos(th)**2)*F2*2*np.pi*np.sin(th)
    return r_e**2*np.trapezoid(f, th)

mu = {}
for E in energies:
    sig_inc = kn_sigma(E)
    lam = 12.398/(E*1000)
    sig_coh = rayleigh_sigma(lam)
    mu[E] = dict(inc=NA_e*sig_inc, coh=N_mol*sig_coh, pair=MU_PAIR_CM, pe=MU_PE_CM)
    print(E, {k:round(v,6) for k,v in mu[E].items()}, "total", sum(mu[E].values()))
