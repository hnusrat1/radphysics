import numpy as np, scipy.integrate as si
xx=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
def interpcol(E,col):return np.exp(np.interp(np.log(E),np.log(xx[:,0]),np.log(np.maximum(xx[:,col],1e-30))))
def comp_fraction(E):
 a=E/.51099895
 def f(m):
  r=1/(1+a*(1-m)); return r*r*(r+1/r-(1-m*m))
 den=si.quad(lambda m:f(m),-1,1)[0];num=si.quad(lambda m:f(m)*(1-1/(1+a*(1-m))),-1,1)[0]
 return num/den
edges=np.array([0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
sp=np.array([0.00583197873298,0.0341428345877,0.0384922608466,0.0336270560045,0.0424974277929,0.0281875542487,0.0210803700289,0.0177019477226,0.0152712058974,0.0144458495229,0.0133465632665,0.0132311922565,0.0132480444293,0.0126415690969,0.0122800649273,0.0115773632354,0.0112026166062,0.0106605229426,0.0105824080578,0.0102302833477,0.0102721938523,0.0102918986805,0.0101312609857,0.00730537131813,0.00457384979463,0.00462005535177,0.00296891086246,0])
Ktr=0
for E,F in [(1.1732,.36100922889),(1.3325,.368159899153)]+[(float((a+b)/2),float(sp[i])) for i,(a,b) in enumerate(zip(edges[:-1],edges[1:]))]:
 if not F or E==0:continue
 inc=interpcol(E,2);photo=interpcol(E,3);pair=interpcol(E,4)+interpcol(E,5)
 mutr=inc*comp_fraction(E)+photo+pair*max(0,(E-1.022)/E)
 Ktr+=F*E*mutr
print(Ktr, Ktr*1.602176634e-10, 'with .003 rad',Ktr*.997*1.602176634e-10)
