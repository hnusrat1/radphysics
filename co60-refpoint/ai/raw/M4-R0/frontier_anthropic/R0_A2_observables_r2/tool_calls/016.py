
ana = r'''
import sys, pickle, numpy as np
sys.path.insert(0,'/tmp')
import phys, mc as M
from phys import mu_inc, mu_pe, mu_pair, mu_coh, mu_tot_fn, mec2, re, NeW, NeAir, es
from scipy.integrate import quad
t=pickle.load(open('/tmp/tallies.pkl','rb'))
N=t['N']; V=np.pi*1.0**2*0.2; Vslab=np.pi*1.0**2*0.1
Om=M.OMEGA; r0=105.0
Fn=Om*r0**2/(N*V)      # TL->fluence ratio per Phi0
Fslab=Om*r0**2/(N*Vslab)
C_GY=1.602177e-10       # Gy cm^2 per (MeV/g * cm^-2 ... ) conversion used as: value*E*mu/rho
ELINES=np.array([1.1732,1.3325]); WL=np.array([0.5,0.5])
# ---------- primary (analytic) ----------
def prim_line_ratio(E):
    mu=float(mu_tot_fn(np.array([E]))[0])
    zz=np.linspace(4.9,5.1,2001)
    f=np.exp(-mu*zz)*(r0/(100+zz))**2
    return np.trapz(f,zz)/0.2
R_line=np.array([0.5*prim_line_ratio(E) for E in ELINES])
# primary angular: all in cos bin 9 (cos>0.997)
# ---------- scattered MC ----------
EC=(np.arange(M.NEB)+0.5)*M.DE
Rs_fine=t['fl_E'][1]*Fn           # per fine bin (ratio to Phi0)
Rp_mc=t['fl_E'][0]*Fn
print('primary analytic',R_line, R_line.sum(),' MC',Rp_mc.sum())
'''
open('/tmp/ana.py','w').write(ana)
import subprocess
print(subprocess.run(['python3','-c','import sys; sys.path.insert(0,"/tmp"); exec(open("/tmp/ana.py").read())'],capture_output=True,text=True))
