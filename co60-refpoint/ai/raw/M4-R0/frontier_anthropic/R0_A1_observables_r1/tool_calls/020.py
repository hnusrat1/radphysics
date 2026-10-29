
anal2 = r'''
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
from phys import re, mc2, ne, Lines, mu_c, mu_pe, mu_pair, mu_coh, mu_tot, ETR, OMEGA
from elec import f_moller, beta2, S_col_BS, S_rad_w, S_res, ne_air
from anal import PE, EE, CE, build
def slowdown(Tg, Ngam_of_T, Sum_w, dens=True):
    """Tg: log grid of T (MeV) ascending. Ngam_of_T: N_gamma(>T) on grid.
    returns Psi_tot, Psi_delta on grid (per Phi0; cm^-2 MeV^-1)"""
    n=len(Tg); Stot=np.array([S_col_BS(T,75e-6,ne,True)+S_rad_w(T) for T in Tg])
    Ssig=np.array([2*np.pi*re**2*mc2/beta2(T) for T in Tg])   # per-electron prefactor (cm^2 MeV)
    Psi=np.zeros(n); Psid=np.zeros(n)
    Qd=np.zeros(n); Nd=np.zeros(n)
    dlnT=np.log(Tg[1]/Tg[0])
    for k in range(n-1,-1,-1):
        Tk=Tg[k]
        j=np.arange(n)
        sel=j[Tg>=2*Tk]
        sel=sel[sel>k]
        if sel.size:
            Tj=Tg[sel]
            integrand=Psi[sel]*Ssig[sel]*f_moller(Tj,Tk)*Tj*dlnT
            Qd[k]=ne*np.sum(integrand)
        if k<n-1:
            Nd[k]=Nd[k+1]+0.5*(Qd[k]+Qd[k+1])*(Tg[k+1]-Tg[k])
        Ntot=Ngam_of_T[k]+Nd[k]
        Psi[k]=Ntot/Stot[k]; Psid[k]=Nd[k]/Stot[k]
    return Psi,Psid,Qd,Nd,Stot
'''
open('/tmp/work/anal2.py','w').write(anal2)
import sys; sys.path.insert(0,'/tmp/work')
import anal2, numpy as np, time
t=time.time()
Tg=np.geomspace(0.01,1.4,400)
Ngam=np.zeros_like(Tg)  # test with dummy
Psi,Psid,Qd,Nd,Stot=anal2.slowdown(Tg,np.exp(-(Tg-0.3)**2/0.01),None)
print(time.time()-t, Psi[:5], Stot[:3], Stot[-3:])
