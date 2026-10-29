from scipy.constants import physical_constants
r_e_scipy = physical_constants['classical electron radius'][0]
print("r_e from scipy:", r_e_scipy)
# compute using that
import numpy as np
mec2 = 0.5109989461
k = 1.1732 / mec2
sigma = 2*np.pi*r_e_scipy**2 * ((1+k)/k**2 * (2*(1+k)/(1+2*k) - np.log(1+2*k)/k) + np.log(1+2*k)/(2*k) - (1+3*k)/(1+2*k)**2)
print("sigma:", sigma)