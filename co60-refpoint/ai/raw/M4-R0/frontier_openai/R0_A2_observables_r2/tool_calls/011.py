import numpy as np, scipy.integrate as si
xx=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
def interpcol(E,col):return np.exp(np.interp(np.log(E),np.log(xx[:,0]),np.log(np.maximum(xx[:,col],1e-30))))
def comp_fraction(E):
 a=E/.51099895
 f=lambda m:(lambda r:r*r*(r+1/r-(1-m*m)))(1/(1+a*(1-m)))
 den=si.quad(lambda m:f(m),-1,1)[0]
 num=si.quad(lambda m:f(m)*(1-1/(1+a*(1-m))),-1,1)[0]
 return num/den
# exact spectra tally primary and scattered
lines=[(1.1732,.36100922889),(1.3325,.368159899153)]
photon_energies=[(float((a+b)/2),float(sp[i])) for i,(a,b) in enumerate(zip(edges[:-1],edges[1:]))]
Ktr=Kcol=0
for E,F in lines:
 inc=interpcol(E,2);photo=interpcol(E,3);pair=interpcol(E,4)+interpcol(E,5)
 mutr=inc*comp_fraction(E)+photo+pair*max(0,(E-1.022)/E)
 Ktr+=F*E*mutr
 Kcol+=F*E*mutr*.997 # assume rad fraction
for E,F in photon_energies:
 if not F: continue
 inc=interpcol(E,2);photo=interpcol(E,3);pair=interpcol(E,4)+interpcol(E,5)
 mutr=inc*comp_fraction(E)+photo+pair*max(0,(E-1.022)/E)
 Ktr+=F*E*mutr
 Kcol+=F*E*mutr*.997
print('MeV/g',Ktr,Kcol,'Gy',Ktr*1.602176634e-10,Kcol*1.602176634e-10, 'g',1-Kcol/Ktr)
