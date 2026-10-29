
code4 = r'''
import numpy as np
from phys import *
# Gauss-Legendre for KN quadrature
_xg,_wg=np.polynomial.legendre.leggauss(400)
def kn_avg(E, func):
    """average over KN distribution of func(eps) at energy E (scalar)"""
    k=E/mc2
    c=_xg
    eps=1/(1+k*(1-c))
    s2=1-c*c
    pdf=eps*eps*(eps+1/eps-s2)
    w=_wg*pdf
    return np.sum(w*func(eps))/np.sum(w)
EG=np.exp(np.linspace(np.log(0.005),np.log(2.5),500))
def _tabs():
    tr=np.zeros_like(EG); col=np.zeros_like(EG); Tm=np.zeros_like(EG)
    for i,E in enumerate(EG):
        Tm[i]=E*kn_avg(E,lambda e:1-e)
        col[i]=E*kn_avg(E,lambda e:(1-e)*(1-Yrad(E*(1-e))))
    return Tm,col
TBL_Tm, TBL_col = _tabs()
def comp_meanT(E): return np.interp(np.log(E),np.log(EG),TBL_Tm)
def comp_colT(E): return np.interp(np.log(E),np.log(EG),TBL_col)
def transfer_coeffs(E):
    """per-process energy transfer (MeV/cm per unit fluence) and collisional deposit"""
    coh,inc,pe,pr,mt = mu_all(E)
    Ec=comp_meanT(E)
    Tc=inc*Ec
    tr_c = Tc
    tr_pe= pe*E
    tr_pa= pr*np.maximum(E-1.022,0)
    Ycol_pair = np.maximum(E-1.022,0)*(1-Yrad(np.maximum((E-1.022)/2,0.01)))
    col_c = inc*comp_colT(E)
    col_pe= pe*E*(1-Yrad(np.maximum(E,0.01)))
    col_pa= pr*Ycol_pair
    return dict(mt=mt,coh=coh,inc=inc,pe=pe,pr=pr,tr_c=tr_c,tr_pe=tr_pe,tr_pa=tr_pa,
                tr=tr_c+tr_pe+tr_pa, col=col_c+col_pe+col_pa, col_c=col_c,col_pe=col_pe,col_pa=col_pa)
'''
open('/tmp/work/deposit.py','w').write(code4)
import deposit
importlib.reload(deposit)
for E in [2.0,1.0,0.5,0.1,0.05]:
    d=deposit.transfer_coeffs(np.array([E]))
    print(E, "K(tr)/E",d['tr'][0]/E, "Kcol/E",d['col'][0]/E, "mu_en-ish", d['col'][0]/E)
