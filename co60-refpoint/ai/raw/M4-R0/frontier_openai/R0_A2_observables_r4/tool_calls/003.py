import numpy as np
from scipy.integrate import quad
re2scale=1
for E in [1.1732,1.3325,1.25]:
 a=E/.51099895
 # KN differential (1/2) r_e2 [ratio^2*(ratio+1/ratio -sin2)] integrated over mu. relative normalized
 def kn(mu):
  q=1/(1+a*(1-mu));return .5*q*q*(q+1/q-(1-mu*mu))
 integral=quad(kn,-1,1)[0]
 avgtransfer=quad(lambda m:kn(m)*(1-1/(1+a*(1-m))),-1,1)[0]/integral
 print(E,integral,avgtransfer, .063*avgtransfer)