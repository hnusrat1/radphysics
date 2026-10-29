import numpy as np, math, time
from collections import Counter
rng=np.random.default_rng(628197)
NA=6.02214076e23
me=0.51099895
# electron density per gram water, 10 e per molecule
ne=NA*10/18.01528
re=2.8179403262e-13 # cm

def sigma_kn(E):
    eps=E/me
    if eps<1e-8: return 8*np.pi*re**2/3
    L=np.log1p(2*eps)
    term=((1+eps)/eps**2)*(2*(1+eps)/(1+2*eps)-L/eps)+L/(2*eps)-(1+3*eps)/(1+2*eps)**2
    return 2*np.pi*re**2*term

def mus(E):
    muc=ne*sigma_kn(E)
    # Simplified non-Compton terms, cm^-1; deliberately small near the incident lines.
    mup=1.2e-5*(max(E,0.005)/1.25)**-2.5
    mur=1.5e-5*(max(E,0.005)/1.25)**-1.3
    mu_pair=0.0
    if E>1.022:
        mu_pair=2.0e-5*((E-1.022)/0.30)**1.4
    return muc,mup,mur,mu_pair
print([(e,mus(e)) for e in [1.1732,1.3325,0.511,0.1]])