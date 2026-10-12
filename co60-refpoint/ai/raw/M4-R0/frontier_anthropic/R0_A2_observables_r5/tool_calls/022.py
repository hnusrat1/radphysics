
pdd = r'''
import numpy as np
from mcana import *
from mcmod import *
import ana as A

def depth_profiles(tal, nh):
    Pphi0 = A.primary_fluence_per_phi0()
    zc = np.linspace(0,10,101); zm=0.5*(zc[:-1]+zc[1:])
    Vb = (2*LAT)**2*0.1
    Ks = tal['zb_kcol']/nh/Vb            # MeV/g per src (rho=1), scattered
    # primary: per src
    Kp = np.zeros(100)
    for E in LINES:
        mu=mu_tot(E); mue=mu_en_all(E)
        Kp += 0.5*Phi0_src*np.exp(-mu*zm)*(105.0/(100.0+zm))**2*E*mue
    Ktot = (Ks+Kp)/Phi0_src              # per Phi0 (MeV cm2/g)
    return zm, Ktot, Ks, Kp

def dose_from_kernel(zm, Ktot, cF, dF, Psi, Psi_scale, C_src_w, Tgrid, Cgrid):
    pass
'''
open('/tmp/work/pdd.py','w').write(pdd)
import sys; sys.path.insert(0,'/tmp/work')
import importlib, pdd as P
importlib.reload(P)
zm,Ktot,Ks,Kp = P.depth_profiles(tal, tal['nh'])
for i in [0,2,4,5,6,8,10,15,20,30,40,50,60,80,99]:
    print(round(zm[i],2), round(Ktot[i]*1.602e-10*1e12,4), "scat frac", round(Ks[i]/(Ks[i]+Kp[i]),3))
# consistency with scoring-volume K at 5 cm
print("V-based Kcol_sc/Phi0 *1e12:", tal['Kcol_sc']/tal['nh']/Phi0_src*1.602e-10*1e12, " profile scatt at 5cm bin:", Ks[50]/Phi0_src*1.602e-10*1e12)
