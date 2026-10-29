
test=r'''
import numpy as np
from sim import *
E=np.array([1.25])
c=np.linspace(-1,1,200001)
for Ev in [1.25,0.5,0.1]:
    q=KN_q(np.full_like(c,Ev),c)
    print(Ev, np.trapezoid(q*2*np.pi,c), sig_KN(Ev))
    qr=rayl_q(np.full_like(c,Ev),c)
    print(' rayl norm', np.trapezoid(qr*2*np.pi,c), sig_R(np.array([Ev])))
'''
open('t2.py','w').write(test)
r=subprocess.run(['python','t2.py'],capture_output=True,text=True,cwd='.'); print(r.stdout,r.stderr[-2000:])
