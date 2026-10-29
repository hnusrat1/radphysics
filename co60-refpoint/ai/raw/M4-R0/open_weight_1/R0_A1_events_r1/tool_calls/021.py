# Let me write the simulation step by step, testing each piece

import numpy as np
from numpy.random import default_rng

rng = default_rng(42)

# === Cross sections ===
mec2 = 0.5110034
re = 2.8179403262e-13
nel_cm3 = 6.02214076e23 * 10 / 18.01528

def sigma_KN(E):
    a = E / mec2
    term = ((1+a)/a**2 * (2*(1+a)/(1+2*a) - np.log(1+2*a)/a) +
            np.log(1+2*a)/(2*a) - (1+3*a)/(1+2*a)**2)
    return 2 * np.pi * re**2 * term

print("testing sigma_KN...")
s1 = sigma_KN(1.1732)
s2 = sigma_K(1.3325)
print(f"sigma_KN(1.1732)={s1:.4e}, sigma_KN(1.3325)={s2:.4e}")