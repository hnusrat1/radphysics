
import subprocess
t=r'''
import numpy as np
from chordlib import chord
p=np.array([[0,0,0.0],[0.5,0,0],[0,0,0],[2,0,0]]); d=np.array([[0,0,1.0],[0,0,1.0],[0.1,0,0.995],[0,0,1.0]])
d=d/np.linalg.norm(d,axis=1)[:,None]
print(chord(p,d,np.array([30,30,30,30.])))
p2=np.array([[0,0,4.0],[0,0,-1.0]]); print(chord(p2,np.array([[0,0,1.0],[0,0,1.0]]),np.array([30.,30.])))
# random test of primary-type rays
rng=np.random.default_rng(1)
n=400000
a=rng.uniform(-0.05,0.05,n); b=rng.uniform(-0.05,0.05,n)
d=np.stack([a,b,np.ones(n)],1); d/=np.linalg.norm(d,axis=1)[:,None]
src=np.array([0,0,-100.])
entry=src+d*(100/d[:,2])[:,None]
ch=chord(entry,d,np.full(n,1e3))
print('mean chord', ch.mean(), 'frac hit', (ch>0).mean(), 'expected frac', np.pi/(105**2)/0.00998)
'''
open('t4.py','w').write(t)
r=subprocess.run(['python','t4.py'],capture_output=True,text=True,cwd='.'); print(r.stdout,r.stderr[-2000:])
