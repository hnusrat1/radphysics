
chk = r'''
import sys, numpy as np
sys.path.insert(0,'/tmp/work')
from phys import *
MEC2=M_E
def dsig(Ep, eps):
    tau = Ep/MEC2; beta2 = tau*(tau+2)/(tau+1)**2
    x = eps/Ep
    F = 1/x**2 + 1/(1-x)**2 + (tau/(tau+1))**2 - (2*tau+1)/(tau+1)**2/(x*(1-x))
    K = 2*np.pi*RE**2*MEC2/beta2
    return K*F/Ep**2
NEL = NA*ZA_W
for Ep in [0.1,0.5,1.0]:
    e = np.geomspace(0.01, Ep/2, 20000)
    I = np.trapezoid(dsig(Ep,e), e)
    # Rutherford-like check: 1/eps^2 coefficient
    print('Ep',Ep,'n*int dsig (per cm)', NEL*I, ' expected ~ 2pi re^2 mc2/beta2 * (1/D-1/Tmax)', NEL*2*np.pi*RE**2*MEC2/(Ep*(Ep+2*MEC2)/(Ep+MEC2)**2)*(1/0.01-2/Ep))
    # energy loss to deltas above 10 keV
    eL = np.trapezoid(e*dsig(Ep,e), e)*NA*ZA_W
    print('  energy loss to deltas >10keV (MeV/cm):', eL)
'''
open('/tmp/work/chk.py','w').write(chk)
import subprocess
r=subprocess.run("cd /tmp/work && python chk.py",shell=True,capture_output=True,text=True)
print(r.stdout, r.stderr[-2000:])
