import math
re=2.8179403262e-13 # cm
me=0.51099895
NA=6.02214076e23
# KN total sigma
for E in [1.1732,1.3325,1.0,.8,.5,.3,.2,.1,.05,.02]:
 x=E/me
 sig=2*math.pi*re**2*((1+x)/x**2*(2*(1+x)/(1+2*x)-math.log(1+2*x)/x)+math.log(1+2*x)/(2*x)-(1+3*x)/(1+2*x)**2)
 mu=sig* (10/18*NA)*1 # e/cm3 = 10 electrons /18g * NA
 print(E, sig/1e-24, mu)
