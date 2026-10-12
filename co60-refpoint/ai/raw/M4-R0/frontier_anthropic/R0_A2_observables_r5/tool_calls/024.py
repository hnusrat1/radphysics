
kern = r'''
import numpy as np
from mcana import *
from mcmod import *
def kernel_from_sources(C_nc, cF, dF, zbin=0.1, xmin=-2.0, xmax=12.0, ns=600):
    """C_nc: (NFINE,10) production counts per src (weights) ; cF centers of fine bins (MeV)
       returns kernel on grid x (bin centres), normalised to 1 per MeV deposited"""
    xg = np.arange(xmin, xmax+zbin/2, zbin); xc=0.5*(xg[:-1]+xg[1:]) if False else xg
    edges = xg
    kern = np.zeros(len(edges)-1)
    cen = np.array([-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9])
    Tgrid = cF
    Rtab = CSDA(np.maximum(Tgrid,0.01))
    tot_energy=0.0
    for i,T in enumerate(Tgrid):
        if T<0.0105: continue
        for j,c in enumerate(cen):
            w = C_nc[i,j]
            if w<=0: continue
            R = CSDA(T)
            s = np.linspace(0,R,ns+1); sm=0.5*(s[1:]+s[:-1]); ds=s[1]-s[0]
            # energy at path position: R(E)=R-s
            Rrem = np.maximum(R-sm,1e-9)
            # invert CSDA: E such that CSDA(E)=Rrem
            Eloc = np.exp(np.interp(np.log(Rrem), np.log(RANGE), np.log(TE)))
            dep = S_col(Eloc)*ds      # MeV/(g/cm2) per segment -> energy deposited (per unit rho)
            x = c*sm
            # bin into depth bins
            idx = np.floor((x-xmin)/zbin).astype(int)
            ok=(idx>=0)&(idx<len(kern))
            np.add.at(kern, idx[ok], w*dep[ok])
            tot_energy += w*dep.sum()
    return xg, kern/ (zbin) , tot_energy
'''
open('/tmp/work/kern.py','w').write(kern)
import kern as K
importlib.reload(K)
out_p=None
# production counts per src (all electrons, scattered+primary), fine bins x cos
Cnc = (tal['q_fine_all_cos']/tal['nh'])
print(Cnc.shape, "total electrons per src (scat only)", tal['q_fine_all_cos'].sum()/tal['nh'])
