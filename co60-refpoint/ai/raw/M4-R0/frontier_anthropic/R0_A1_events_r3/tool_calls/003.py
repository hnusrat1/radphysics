
import math
MEC2 = 0.51099895; RE = 2.8179403e-13; NE_WATER = 3.3428e23
def sigma_KN(E):
    k = E/MEC2
    t1 = (1+k)/k**2*(2*(1+k)/(1+2*k) - math.log(1+2*k)/k)
    t2 = math.log(1+2*k)/(2*k)
    t3 = -(1+3*k)/(1+2*k)**2
    return 2*math.pi*RE**2*(t1+t2+t3)
for E in [0.5,1.0,1.1732,1.25,1.3325]:
    print(E, sigma_KN(E)*1e24, NE_WATER*sigma_KN(E))
