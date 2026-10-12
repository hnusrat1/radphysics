
import numpy as np, time
rng=np.random.default_rng(1)
re2=7.9407877e-26; ne=6.02214e23*10/18.015  # electrons per cm^3 (water, rho=1)
def sKN(E):
    k=E/0.51099895; L=np.log(1+2*k)
    a=(1+k)/k**2*(2*(1+k)/(1+2*k)-L/k)+L/(2*k)-(1+3*k)/(1+2*k)**2
    return 2*np.pi*re2*a
def mu_pe(E): return 2.7e-3*(0.1/E)**3
def mu_coh(E): return 1.0e-4*(2.0/E)**1.33
def mu_pair(E):
    f=np.clip((E-1.022)/0.978,0,None); return 2.6e-3*f**1.5
def mu_inc(E): return sKN(E)*ne
def mu_tot(E): return mu_inc(E)+mu_pe(E)+mu_coh(E)+mu_pair(E)
print("incoh 2MeV",mu_inc(2.0),"total",mu_tot(2.0),"NIST 4.942e-2")
print("incoh 0.1",mu_inc(0.1),"tot 0.1",mu_tot(0.1))
# energy grid tables
Egrid=np.logspace(np.log10(0.01),np.log10(2.0),400)
mugrid=np.log(mu_tot(Egrid))
def mu_of(E): return np.exp(np.interp(np.log(E),np.log(Egrid),mugrid))
