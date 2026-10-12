import scipy.integrate as si, numpy as np
m=.51099895
for E in [1.17,1.33]:
 def f(mu):
  ep=E/(1+E/m*(1-mu)); r=ep/E
  return r*r*(r+1/r-(1-mu*mu))
 sig=si.quad(f,-1,1)[0]
 av=si.quad(lambda mu:f(mu)*(E-E/(1+E/m*(1-mu))),-1,1)[0]/sig
 print(E,av,av/E, .063*av/E)
