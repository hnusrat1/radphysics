
trap=np.trapezoid
def kn_tr_frac(E):
    k=E/me
    c=np.linspace(-1,1,40001)
    eps=1/(1+k*(1-c))
    f=eps**2*(eps+1/eps-(1-c**2))
    sig=np.pi*re**2*trap(f,c)
    sigf=np.pi*re**2*trap(f*(1-eps),c)
    return sig,sigf
for E in [0.1,0.5,1.0,1.1732,1.25,1.3325]:
    s,sf=kn_tr_frac(E)
    print(E, round(kn_sigma(E)*ne_w,5), round(s*ne_w,5), round(sf*ne_w,5), 'tab',mu_tab(E), 'resid', round(mu_tab(E)-kn_sigma(E)*ne_w,5), 'frac',round(sf/s,4))
