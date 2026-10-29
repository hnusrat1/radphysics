
import numpy as np
from scipy.integrate import quad

r_e2 = 2.8179403262e-13**2  # cm^2
mec2 = 0.5109989461  # MeV
two_pi = 2*np.pi

def kn_diff_xs(E, costh):
    # dsigma/dOmega per electron, unpolarized Klein-Nishina
    # E photon energy MeV, costh = cos of photon scattering angle
    k = E/mec2
    Ep = E/(1.0 + k*(1.0-costh))
    ratio = Ep/E
    # bracket: E/E' + E'/E - sin^2 theta = 1/ratio + ratio - (1-costh^2)
    bracket = 1.0/ratio + ratio - (1.0 - costh*costh)
    return (r_e2/2.0)*ratio*ratio*bracket

def kn_total(E):
    f = lambda th: kn_diff_xs(E, np.cos(th))*np.sin(th)
    val, err = quad(f, 0, np.pi, epsabs=1e-30, limit=200)
    return two_pi*val

for E in [0.01,0.05,0.1,0.2,0.5,1.0,1.25,1.5,2.0]:
    s = kn_total(E)
    print(E, s, s/ ( (8*np.pi/3)*r_e2 ), 'barn', s*1e24)
