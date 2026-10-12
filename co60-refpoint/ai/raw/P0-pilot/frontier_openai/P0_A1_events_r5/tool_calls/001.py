import scipy.integrate as si, math
re=2.8179403262e-13
for E in [0.511,1,2]:
 x=E/.51099895
 # total KN formula
 sig=2*math.pi*re**2*((1+x)/x**2*(2*(1+x)/(1+2*x)-math.log(1+2*x)/x)+math.log(1+2*x)/(2*x)-(1+3*x)/(1+2*x)**2)
 mu=sig*3.343e23
 print(E,sig/1e-24,mu)
