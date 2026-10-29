
import subprocess
t=r'''
import numpy as np
from phys import *
import sim as S
rng=S.rng
n=300000
E=np.full(n,1.25)
eps=S.KN_eps(E)
cc=1-(1/eps-1)/(E/me)
print('mean cos', cc.mean(), 'mean eps', eps.mean())
# compare to KN: <cos> for 1.25 MeV ~ ?
c=np.linspace(-1,1,400001)
q=S.KN_q(np.full_like(c,1.25),c)
print('KN mean cos (numeric)', np.trapezoid(q*c*2*np.pi,c))
print('KN eps hist vs q: ', np.histogram(cc,bins=5,range=(-1,1))[0]/n)
h=np.histogram(cc,bins=5,range=(-1,1))[0]/n
ce=np.linspace(-1,1,6)
num=[np.trapezoid((q*2*np.pi)[(c>=ce[i])&(c<=ce[i+1])],c[(c>=ce[i])&(c<=ce[i+1])]) for i in range(5)]
print('expected', np.round(num,4))
'''
open('t8.py','w').write(t)
r=subprocess.run(['python','t8.py'],capture_output=True,text=True,cwd='.'); print(r.stdout,r.stderr[-2000:])
