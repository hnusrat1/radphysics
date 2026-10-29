# Continue from the previous state? Can't - separate cells. Let me run everything together.
import numpy as np, math, time
from collections import Counter

rng = np.random.default_rng(12345)
me=0.51099895; re=2.8179403262e-13; Ne=3.343e23

def kn_f(mu,E):
    a=E/me; r=1.0/(1.0+a*(1.0-mu))
    return r*r*(1.0/r+r-(1.0-mu*mu))

Egrid=np.logspace(np.log10(0.001),np.log10(10.0),300)
sig_in = np.empty(300)
|for j,E in enumerate(Egrid):
    mus=np.linspace(-1,1,3001)
    sig_grid[j]=np.pi*re**2*np.trapezoid(kn_f(mus,E),mus)