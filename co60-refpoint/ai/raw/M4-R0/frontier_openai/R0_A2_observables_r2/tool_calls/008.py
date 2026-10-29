import scipy.integrate as si, math
for E in (1.1732,1.3325):
 a=E/.51099895
 def dcs(mu):
  r=1/(1+a*(1-mu)); return r*r*(r+1/r-(1-mu*mu))
 tot=si.quad(lambda m:dcs(m),-1,1)[0]
 frac=si.quad(lambda m:dcs(m)*(1-1/(1+a*(1-m))),-1,1)[0]/tot
 print(E,frac)
