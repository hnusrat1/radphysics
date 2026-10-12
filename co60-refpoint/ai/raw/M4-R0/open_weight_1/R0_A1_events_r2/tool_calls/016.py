import numpy as np
import time

# Constants
r_e = 2.8179403227e-13  # cm
me = 0.5109989461  # MeV
NA = 6.02214076e23
M_H2O = 18.015
rho = 1.0
N_e = 10
E1 = 1.1732
E2 = 1.3325

# Geometry
XMIN, XMAX = -15.0, 15.0
YMIN, YMAX = -15.0, 15.0
ZMIN, ZMAX = 0.0, 30.0
SV_R = 1.0
SV_ZLO = 4.9
SV_ZHI = 5.1

# Cross sections
def sigma_KN(E):
    k = E / me
    term = ((1+k)/k**2 * (2*(1+k)/(1+2*k) - np.log(1+2*k)/k) +
            np.log(1+2*k)/(2*k) - (1+3*k)/(1+2*k)**2)
    # fix syntax
    pass

# Let's use a simpler known table for water to avoid complex coding.
# I'll use precomputed mass attenuation coefficients from NIST.
# Since I cannot access internet, I will use approximate values.

# Actually, let's just compute using the correct Klein-Nishina formula.

print("Let me compute correctly.")
k = 1.1732 / me
term = ((1+k)/k**2 * (2*(1+k)/(1+2*k) - np.log(1+2*k)/k + np.log(1+2*k)/(2*k) - (1+3*k)/(1+2*k)**2
print(term)