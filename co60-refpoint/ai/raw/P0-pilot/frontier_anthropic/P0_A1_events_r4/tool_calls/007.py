
xs_code = '''
import numpy as np
re2=7.9407877e-26; ne=6.02214e23*10/18.015
def sKN(E):
    k=E/0.51099895; L=np.log(1+2*k)
    a=(1+k)/k**2*(2*(1+k)/(1+2*k)-L/k)+L/(2*k)-(1+3*k)/(1+2*k)**2
    return 2*np.pi*re2*a
def fbind(E): return 1-0.046*np.clip(E/2.0,1e-3,None)**0.42
def mu_pe(E): return 2.7e-3*(0.1/E)**3
def mu_coh(E): return 1.0e-4*(2.0/E)**1.33
def mu_inc(E): return sKN(E)*ne*fbind(E)
def mu_pair(E):
    f=np.clip((E-1.022)/0.978,0,None); return 2.66e-3*f**1.5
def mu_tot(E): return mu_inc(E)+mu_pe(E)+mu_coh(E)+mu_pair(E)
Egrid=np.logspace(np.log10(0.01),np.log10(2.0),600)
mugrid=np.log(mu_tot(Egrid))
def mu_of(E): return np.exp(np.interp(np.log(E),np.log(Egrid),mugrid))
'''
with open('/tmp/work/xs.py','w') as f: f.write(xs_code)
import importlib, sys
sys.path.insert(0,'/tmp/work')
import xs
print(xs.mu_tot(2.0), xs.mu_inc(2.0), xs.mu_tot(0.1), xs.mu_tot(0.5), xs.mu_tot(1.0))
