
import subprocess
test2=r'''
import numpy as np, simtl
Ns=400000
C=simtl.run(Ns)
tl=simtl.TL
print('primary-segment TL per src photon:', tl[0].sum()/Ns)
print('all-segment TL per src photon:', sum(t.sum() for t in tl)/Ns)
for ty in [0,1,2,3]:
    m=C[:,0]==ty
    print(ty, C[m,1].sum()/Ns)
'''
open('t3.py','w').write(test2)
r=subprocess.run(['python','t3.py'],capture_output=True,text=True,cwd='.'); print(r.stdout,r.stderr[-2000:])
